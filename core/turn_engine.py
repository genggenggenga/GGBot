"""Lightweight, explicit turn-level state machine for Agent execution."""
import inspect
from typing import Awaitable, Callable, Dict, Set, Union

from core.agent_models import (
    ConfirmationStatus,
    DialogueState,
    ExecutionState,
    HandoffPackage,
    Transition,
    TurnContext,
    TurnEvent,
    TurnEventType,
)
from core.state_store import StateStore


class InvalidTransitionError(ValueError):
    """Raised when a handler requests a transition not allowed by the graph."""


HandlerResult = Union[TurnEvent, Awaitable[TurnEvent]]
StateHandler = Callable[[TurnContext], HandlerResult]


_PAUSED_STATES = {
    ExecutionState.CLARIFYING,
    ExecutionState.AWAITING_CONFIRMATION,
}
_TERMINAL_STATES = {
    ExecutionState.COMPLETED,
    ExecutionState.FAILED,
}

_EVENT_TRANSITIONS = {
    (ExecutionState.UNDERSTANDING, TurnEventType.CLARIFICATION_REQUIRED): (
        ExecutionState.CLARIFYING
    ),
    (ExecutionState.UNDERSTANDING, TurnEventType.UNDERSTANDING_ACCEPTED): (
        ExecutionState.ROUTING
    ),
    (ExecutionState.ROUTING, TurnEventType.ACTION_REJECTED): (
        ExecutionState.RESPONDING
    ),
    (ExecutionState.ROUTING, TurnEventType.ROUTED_TO_KNOWLEDGE): (
        ExecutionState.RETRIEVING
    ),
    (ExecutionState.ROUTING, TurnEventType.ROUTED_TO_ACTION): (
        ExecutionState.ACTING
    ),
    (ExecutionState.RETRIEVING, TurnEventType.AGENT_FAILED): (
        ExecutionState.FAILED
    ),
    (ExecutionState.RETRIEVING, TurnEventType.SLOTS_MISSING): (
        ExecutionState.CLARIFYING
    ),
    (ExecutionState.RETRIEVING, TurnEventType.AGENT_COMPLETED): (
        ExecutionState.RESPONDING
    ),
    (ExecutionState.ACTING, TurnEventType.AGENT_FAILED): (
        ExecutionState.FAILED
    ),
    (ExecutionState.ACTING, TurnEventType.SLOTS_MISSING): (
        ExecutionState.CLARIFYING
    ),
    (ExecutionState.ACTING, TurnEventType.WRITE_CONFIRMATION_REQUIRED): (
        ExecutionState.AWAITING_CONFIRMATION
    ),
    (ExecutionState.ACTING, TurnEventType.AGENT_COMPLETED): (
        ExecutionState.RESPONDING
    ),
    (ExecutionState.RESPONDING, TurnEventType.RESPONSE_READY): (
        ExecutionState.COMPLETED
    ),
}


def _derive_allowed_transitions() -> Dict[ExecutionState, Set[ExecutionState]]:
    allowed = {state: set() for state in ExecutionState}
    for (state, _), target in _EVENT_TRANSITIONS.items():
        allowed[state].add(target)
    return allowed


_ALLOWED_TRANSITIONS = _derive_allowed_transitions()


def supported_event_types() -> Set[TurnEventType]:
    """Return event types that have at least one declared transition edge."""
    return {event for _, event in _EVENT_TRANSITIONS}


def resolve_next_state(
    current: ExecutionState,
    event: TurnEventType,
) -> ExecutionState:
    """Resolve an event into its next execution state."""
    next_state = _EVENT_TRANSITIONS.get((current, event))
    if next_state is None:
        raise InvalidTransitionError(
            f"event not allowed: {current.value} + {event.value}",
        )
    return next_state


def transition_from_event(
    current: ExecutionState,
    event: TurnEvent,
) -> Transition:
    """Build the internal Transition for applying a semantic TurnEvent."""
    return Transition(
        next_state=resolve_next_state(current, event.type),
        event=event.type,
        dialogue_updates=event.dialogue_updates,
        observations=event.observations,
        response=event.response,
        reason=event.reason or event.type.value,
    )


def validate_transition(
    current: ExecutionState,
    target: ExecutionState,
    event: TurnEventType,
) -> None:
    """Validate a state edge before applying any context changes."""
    if target not in _ALLOWED_TRANSITIONS[current]:
        raise InvalidTransitionError(
            f"illegal transition: {current.value} -> {target.value}",
        )
    expected = resolve_next_state(current, event)
    if expected != target:
        raise InvalidTransitionError(
            "event transition mismatch: "
            f"{current.value} + {event.value} -> {expected.value}, "
            f"got {target.value}",
        )


def apply_transition(context: TurnContext, transition: Transition) -> TurnContext:
    """Apply one validated transition and return a new runtime context."""
    validate_transition(
        context.execution_state,
        transition.next_state,
        transition.event,
    )

    state_data = context.dialogue_state.model_dump()
    state_data.update(transition.dialogue_updates)
    if transition.dialogue_updates and "state_version" not in transition.dialogue_updates:
        state_data["state_version"] = context.dialogue_state.state_version + 1
    dialogue_state = DialogueState.model_validate(state_data)

    observations = [*context.observations, *transition.observations]
    handoff = context.handoff
    if transition.next_state == ExecutionState.FAILED:
        handoff = HandoffPackage(
            reason=transition.reason,
            active_intent=dialogue_state.active_intent,
            slots=dict(dialogue_state.slots),
            observations=observations,
            failed_state=context.execution_state,
        )

    return context.model_copy(
        update={
            "dialogue_state": dialogue_state,
            "execution_state": transition.next_state,
            "state_history": [
                *context.state_history,
                transition.next_state,
            ],
            "observations": observations,
            "step_count": context.step_count + 1,
            "response": transition.response,
            "handoff": handoff,
        },
        deep=True,
    )


class TurnEngine:
    """Run registered handlers until completion, failure, or user input is needed."""

    def __init__(self, state_store: StateStore, max_steps: int = 6) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self._state_store = state_store
        self._max_steps = max_steps
        self._handlers: Dict[ExecutionState, StateHandler] = {}

    def register(self, state: ExecutionState, handler: StateHandler) -> None:
        if state in _TERMINAL_STATES:
            raise ValueError(f"cannot register handler for terminal state: {state.value}")
        self._handlers[state] = handler

    def fork(self) -> "TurnEngine":
        """Create an isolated handler registry over the same state store."""
        return TurnEngine(self._state_store, max_steps=self._max_steps)

    async def load_context(self, user_id: str, conv_id: str) -> TurnContext:
        """Restore persisted dialogue state and derive its resumable execution state."""
        dialogue_state = await self._state_store.load(user_id, conv_id)
        if dialogue_state is None:
            dialogue_state = DialogueState()
            execution_state = ExecutionState.UNDERSTANDING
        else:
            execution_state = self.resume_state(dialogue_state)
        return TurnContext(
            user_id=user_id,
            conv_id=conv_id,
            dialogue_state=dialogue_state,
            execution_state=execution_state,
            state_history=[execution_state],
        )

    @staticmethod
    def resume_state(dialogue_state: DialogueState) -> ExecutionState:
        """Derive where a later turn should resume from persisted business state."""
        if dialogue_state.missing_slots:
            return ExecutionState.CLARIFYING
        if dialogue_state.pending_action is not None:
            if dialogue_state.confirmation_status == ConfirmationStatus.PENDING:
                return ExecutionState.AWAITING_CONFIRMATION
            if dialogue_state.confirmation_status == ConfirmationStatus.CONFIRMED:
                return ExecutionState.ACTING
        if dialogue_state.confirmation_status == ConfirmationStatus.REJECTED:
            return ExecutionState.RESPONDING
        return ExecutionState.ROUTING

    async def run(self, context: TurnContext) -> TurnContext:
        """Execute a bounded state loop and persist state after every step."""
        if context.execution_state in _TERMINAL_STATES:
            await self._save(context)
            return context

        current = context
        first_step = True
        while current.execution_state not in _TERMINAL_STATES:
            if current.execution_state in _PAUSED_STATES and not first_step:
                break
            if current.step_count >= self._max_steps:
                current = self._fail(current, "max_steps_exceeded")
                await self._save(current)
                return current

            handler = self._handlers.get(current.execution_state)
            if handler is None:
                current = self._fail(
                    current,
                    f"handler_not_registered:{current.execution_state.value}",
                )
                await self._save(current)
                return current

            try:
                event = handler(current)
                if inspect.isawaitable(event):
                    event = await event
                transition = transition_from_event(
                    current.execution_state,
                    event,
                )
                current = apply_transition(current, transition)
            except InvalidTransitionError:
                current = self._fail(current, "invalid_transition")
            except Exception as ex:
                current = self._fail(
                    current,
                    f"handler_failed:{type(ex).__name__}",
                )

            await self._save(current)
            first_step = False

        return current

    async def _save(self, context: TurnContext) -> None:
        await self._state_store.save(
            context.user_id,
            context.conv_id,
            context.dialogue_state,
        )

    @staticmethod
    def _fail(context: TurnContext, reason: str) -> TurnContext:
        handoff = HandoffPackage(
            reason=reason,
            active_intent=context.dialogue_state.active_intent,
            slots=dict(context.dialogue_state.slots),
            observations=list(context.observations),
            failed_state=context.execution_state,
        )
        return context.model_copy(
            update={
                "execution_state": ExecutionState.FAILED,
                "response": "自动处理未完成，已整理信息供人工客服继续处理。",
                "handoff": handoff,
            },
            deep=True,
        )
