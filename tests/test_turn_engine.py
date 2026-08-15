import pytest

from core.agent_models import (
    ConfirmationStatus,
    DialogueState,
    ExecutionState,
    PendingAction,
    Transition,
    TurnContext,
    TurnEvent,
    TurnEventType,
)
from core.state_store import InMemoryStateStore
from core.turn_engine import (
    InvalidTransitionError,
    TurnEngine,
    apply_transition,
    supported_event_types,
    transition_from_event,
)


def raw_transition(
    next_state: ExecutionState,
    reason: str = "test",
    event: TurnEventType = TurnEventType.RESPONSE_READY,
) -> Transition:
    return Transition(next_state=next_state, event=event, reason=reason)


@pytest.mark.asyncio
async def test_normal_execution_completes_and_persists_state():
    store = InMemoryStateStore()
    engine = TurnEngine(store)
    engine.register(
        ExecutionState.UNDERSTANDING,
        lambda _: TurnEvent(
            type=TurnEventType.UNDERSTANDING_ACCEPTED,
        ),
    )
    engine.register(
        ExecutionState.ROUTING,
        lambda _: TurnEvent(
            type=TurnEventType.ROUTED_TO_KNOWLEDGE,
        ),
    )
    engine.register(
        ExecutionState.RETRIEVING,
        lambda _: TurnEvent(
            type=TurnEventType.AGENT_COMPLETED,
            response="处理完成",
        ),
    )
    engine.register(
        ExecutionState.RESPONDING,
        lambda _: TurnEvent(
            type=TurnEventType.RESPONSE_READY,
            response="处理完成",
            reason="response_ready",
        ),
    )

    result = await engine.run(
        TurnContext(
            user_id="u1",
            conv_id="c1",
            dialogue_state=DialogueState(active_intent="order_query"),
        ),
    )

    assert result.execution_state == ExecutionState.COMPLETED
    assert result.response == "处理完成"
    assert result.step_count == 4
    persisted = await store.load("u1", "c1")
    assert persisted is not None
    assert persisted.active_intent == "order_query"


@pytest.mark.asyncio
async def test_engine_accepts_turn_event_handler_output():
    store = InMemoryStateStore()
    engine = TurnEngine(store)
    engine.register(
        ExecutionState.UNDERSTANDING,
        lambda _: TurnEvent(
            type=TurnEventType.UNDERSTANDING_ACCEPTED,
            dialogue_updates={"active_intent": "order_query"},
        ),
    )
    engine.register(
        ExecutionState.ROUTING,
        lambda _: TurnEvent(
            type=TurnEventType.ROUTED_TO_KNOWLEDGE,
            dialogue_updates={"last_agent": "knowledge"},
        ),
    )
    engine.register(
        ExecutionState.RETRIEVING,
        lambda _: TurnEvent(
            type=TurnEventType.AGENT_COMPLETED,
            response="查询完成",
        ),
    )
    engine.register(
        ExecutionState.RESPONDING,
        lambda _: TurnEvent(
            type=TurnEventType.RESPONSE_READY,
            response="查询完成",
        ),
    )

    result = await engine.run(TurnContext(user_id="u1", conv_id="c1"))
    persisted = await store.load("u1", "c1")

    assert result.execution_state == ExecutionState.COMPLETED
    assert result.response == "查询完成"
    assert result.state_history == [
        ExecutionState.UNDERSTANDING,
        ExecutionState.ROUTING,
        ExecutionState.RETRIEVING,
        ExecutionState.RESPONDING,
        ExecutionState.COMPLETED,
    ]
    assert persisted is not None
    assert persisted.active_intent == "order_query"
    assert persisted.last_agent == "knowledge"


def test_illegal_transition_is_rejected_before_context_changes():
    context = TurnContext(user_id="u1", conv_id="c1")

    with pytest.raises(
        InvalidTransitionError,
        match="understanding -> completed",
    ):
        apply_transition(
            context,
            raw_transition(ExecutionState.COMPLETED),
        )

    assert context.execution_state == ExecutionState.UNDERSTANDING
    assert context.step_count == 0


def test_event_transition_must_match_declared_next_state():
    context = TurnContext(user_id="u1", conv_id="c1")

    result = apply_transition(
        context,
        Transition(
            next_state=ExecutionState.ROUTING,
            event=TurnEventType.UNDERSTANDING_ACCEPTED,
            reason="understanding_complete",
        ),
    )

    assert result.execution_state == ExecutionState.ROUTING

    with pytest.raises(
        InvalidTransitionError,
        match="event transition mismatch",
    ):
        apply_transition(
            context,
            Transition(
                next_state=ExecutionState.CLARIFYING,
                event=TurnEventType.UNDERSTANDING_ACCEPTED,
                reason="wrong_next_state",
            ),
        )


def test_transition_from_event_resolves_next_state_and_preserves_payload():
    event = TurnEvent(
        type=TurnEventType.CLARIFICATION_REQUIRED,
        dialogue_updates={
            "active_intent": "refund_request",
            "missing_slots": ["order_id"],
            "required_slots": ["order_id"],
        },
        response="请提供订单号。",
        reason="missing_slot:order_id",
    )

    transition = transition_from_event(
        ExecutionState.UNDERSTANDING,
        event,
    )

    assert transition.next_state == ExecutionState.CLARIFYING
    assert transition.event == TurnEventType.CLARIFICATION_REQUIRED
    assert transition.dialogue_updates == event.dialogue_updates
    assert transition.observations == []
    assert transition.response == "请提供订单号。"
    assert transition.reason == "missing_slot:order_id"


def test_transition_from_event_uses_event_type_as_default_reason():
    transition = transition_from_event(
        ExecutionState.RESPONDING,
        TurnEvent(type=TurnEventType.RESPONSE_READY),
    )

    assert transition.next_state == ExecutionState.COMPLETED
    assert transition.reason == "response_ready"


def test_every_turn_event_type_has_declared_transition():
    assert supported_event_types() == set(TurnEventType)


def test_event_must_be_allowed_for_current_state():
    context = TurnContext(user_id="u1", conv_id="c1")

    with pytest.raises(
        InvalidTransitionError,
        match="event not allowed",
    ):
        apply_transition(
            context,
            Transition(
                next_state=ExecutionState.ROUTING,
                event=TurnEventType.RESPONSE_READY,
                reason="wrong_event",
            ),
        )


@pytest.mark.asyncio
async def test_handler_illegal_transition_becomes_structured_failure():
    engine = TurnEngine(InMemoryStateStore())
    engine.register(
        ExecutionState.UNDERSTANDING,
        lambda _: TurnEvent(type=TurnEventType.RESPONSE_READY),
    )

    result = await engine.run(TurnContext(user_id="u1", conv_id="c1"))

    assert result.execution_state == ExecutionState.FAILED
    assert result.handoff is not None
    assert result.handoff.reason == "invalid_transition"


@pytest.mark.asyncio
async def test_clarifying_pauses_and_can_be_restored_next_turn():
    store = InMemoryStateStore()
    engine = TurnEngine(store)
    engine.register(
        ExecutionState.UNDERSTANDING,
        lambda _: TurnEvent(
            type=TurnEventType.CLARIFICATION_REQUIRED,
            dialogue_updates={
                "active_intent": "refund_request",
                "required_slots": ["order_id"],
                "missing_slots": ["order_id"],
            },
            response="请提供订单号。",
            reason="missing_order_id",
        ),
    )

    result = await engine.run(TurnContext(user_id="u1", conv_id="c1"))
    restored = await engine.load_context("u1", "c1")

    assert result.execution_state == ExecutionState.CLARIFYING
    assert result.response == "请提供订单号。"
    assert restored.execution_state == ExecutionState.CLARIFYING
    assert restored.dialogue_state.missing_slots == ["order_id"]


@pytest.mark.asyncio
async def test_filled_slot_resumes_from_routing_on_the_next_turn():
    store = InMemoryStateStore()
    await store.save(
        "u1",
        "c1",
        DialogueState(
            active_intent="refund_request",
            slots={"order_id": "ORD-1"},
            required_slots=["order_id"],
        ),
    )
    engine = TurnEngine(store)
    engine.register(
        ExecutionState.ROUTING,
        lambda _: TurnEvent(
            type=TurnEventType.ACTION_REJECTED,
            response="已继续处理退款申请。",
        ),
    )
    engine.register(
        ExecutionState.RESPONDING,
        lambda _: TurnEvent(
            type=TurnEventType.RESPONSE_READY,
            response="已继续处理退款申请。",
            reason="response_ready",
        ),
    )

    restored = await engine.load_context("u1", "c1")
    result = await engine.run(restored)

    assert restored.execution_state == ExecutionState.ROUTING
    assert result.execution_state == ExecutionState.COMPLETED
    assert result.response == "已继续处理退款申请。"


@pytest.mark.asyncio
async def test_pending_confirmation_pauses_and_confirmed_action_resumes():
    store = InMemoryStateStore()
    pending = DialogueState(
        active_intent="refund_request",
        slots={"order_id": "ORD-1"},
        pending_action=PendingAction(
            action_id="act-1",
            tool_name="create_refund",
            arguments={"order_id": "ORD-1"},
        ),
        confirmation_status=ConfirmationStatus.PENDING,
    )
    await store.save("u1", "c1", pending)
    engine = TurnEngine(store)

    paused = await engine.load_context("u1", "c1")
    confirmed = pending.model_copy(
        update={"confirmation_status": ConfirmationStatus.CONFIRMED},
    )

    assert paused.execution_state == ExecutionState.AWAITING_CONFIRMATION
    assert engine.resume_state(confirmed) == ExecutionState.ACTING


def test_rejected_action_resumes_to_response():
    state = DialogueState(
        active_intent="refund_request",
        confirmation_status=ConfirmationStatus.REJECTED,
    )

    assert TurnEngine.resume_state(state) == ExecutionState.RESPONDING


@pytest.mark.asyncio
async def test_confirmed_action_continues_on_the_next_turn():
    store = InMemoryStateStore()
    confirmed = DialogueState(
        active_intent="refund_request",
        slots={"order_id": "ORD-1"},
        pending_action=PendingAction(
            action_id="act-1",
            tool_name="create_refund",
            arguments={"order_id": "ORD-1"},
        ),
        confirmation_status=ConfirmationStatus.CONFIRMED,
    )
    await store.save("u1", "c1", confirmed)

    engine = TurnEngine(store)
    engine.register(
        ExecutionState.ACTING,
        lambda _: TurnEvent(
            type=TurnEventType.AGENT_COMPLETED,
            dialogue_updates={
                "pending_action": None,
                "confirmation_status": ConfirmationStatus.NOT_REQUIRED,
            },
            response="退款申请已提交。",
            reason="refund_created",
        ),
    )
    engine.register(
        ExecutionState.RESPONDING,
        lambda context: TurnEvent(
            type=TurnEventType.RESPONSE_READY,
            response=context.response,
            reason="response_ready",
        ),
    )

    restored = await engine.load_context("u1", "c1")
    result = await engine.run(restored)

    assert restored.execution_state == ExecutionState.ACTING
    assert result.execution_state == ExecutionState.COMPLETED
    assert result.response == "退款申请已提交。"
    assert result.dialogue_state.pending_action is None


@pytest.mark.asyncio
async def test_step_limit_fails_with_handoff_package():
    store = InMemoryStateStore()
    engine = TurnEngine(store, max_steps=2)
    engine.register(
        ExecutionState.ROUTING,
        lambda _: TurnEvent(type=TurnEventType.ROUTED_TO_KNOWLEDGE),
    )
    engine.register(
        ExecutionState.RETRIEVING,
        lambda _: TurnEvent(type=TurnEventType.AGENT_COMPLETED),
    )
    context = TurnContext(
        user_id="u1",
        conv_id="c1",
        dialogue_state=DialogueState(
            active_intent="logistics_query",
            slots={"order_id": "ORD-1"},
        ),
        execution_state=ExecutionState.ROUTING,
    )

    result = await engine.run(context)

    assert result.execution_state == ExecutionState.FAILED
    assert result.step_count == 2
    assert result.handoff is not None
    assert result.handoff.reason == "max_steps_exceeded"
    assert result.handoff.active_intent == "logistics_query"
    assert result.handoff.slots == {"order_id": "ORD-1"}


@pytest.mark.asyncio
async def test_handler_failure_creates_structured_handoff():
    store = InMemoryStateStore()
    engine = TurnEngine(store)

    def broken_handler(_: TurnContext) -> TurnEvent:
        raise RuntimeError("private detail")

    engine.register(ExecutionState.UNDERSTANDING, broken_handler)

    result = await engine.run(TurnContext(user_id="u1", conv_id="c1"))

    assert result.execution_state == ExecutionState.FAILED
    assert result.handoff is not None
    assert result.handoff.reason == "handler_failed:RuntimeError"
    assert "private detail" not in result.handoff.reason


@pytest.mark.asyncio
async def test_missing_handler_fails_without_calling_unknown_code():
    result = await TurnEngine(InMemoryStateStore()).run(
        TurnContext(
            user_id="u1",
            conv_id="c1",
            execution_state=ExecutionState.ROUTING,
        ),
    )

    assert result.execution_state == ExecutionState.FAILED
    assert result.handoff is not None
    assert result.handoff.reason == "handler_not_registered:routing"


def test_terminal_handlers_and_invalid_max_steps_are_rejected():
    store = InMemoryStateStore()

    with pytest.raises(ValueError, match="max_steps"):
        TurnEngine(store, max_steps=0)

    engine = TurnEngine(store)
    with pytest.raises(ValueError, match="terminal state"):
        engine.register(
            ExecutionState.COMPLETED,
            lambda _: TurnEvent(type=TurnEventType.AGENT_FAILED),
        )
