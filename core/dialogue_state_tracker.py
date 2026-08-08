"""DialogueStateTracker: deterministic Reducer that merges UnderstandingResult
into the current DialogueState.

Covers:
  - Slot inheritance across turns
  - Slot correction (user explicitly changes a value)
  - Intent switching (user changes goal, only cross-reuse slots survive)
  - Missing slot computation from INTENT_SCHEMAS
  - Tool result merging via observations
"""
import logging
from typing import List, Optional, Set

from core.agent_models import (
    INTENT_SCHEMAS,
    ConfirmationStatus,
    DialogueState,
    Observation,
    PendingAction,
    UnderstandingResult,
    UserAct,
    get_missing_slots,
)

logger = logging.getLogger(__name__)

# Slots that are reusable across different intents
_CROSS_INTENT_SLOTS: Set[str] = {"order_id", "tracking_no"}


class DialogueStateTracker:
    """Deterministic Reducer: (old_state, understanding, observations) -> new_state."""

    def update(
        self,
        state: DialogueState,
        understanding: UnderstandingResult,
        observations: Optional[List[Observation]] = None,
    ) -> DialogueState:
        """Apply understanding and observations to produce a new DialogueState.

        This is a pure function: it does not mutate ``state`` in place.
        Returns a new DialogueState with incremented state_version.
        """
        # Deep copy to avoid mutating the input
        new_slots = dict(state.slots)
        new_completed = list(state.completed_goals)
        new_queued = list(state.queued_goals)
        new_missing: List[str] = []
        new_required: List[str] = []
        new_confirmation = state.confirmation_status
        new_pending = state.pending_action
        new_last_agent = state.last_agent
        new_active = state.active_intent

        # ── 1. Resolve user act ───────────────────────────────────────────────
        user_act = understanding.user_act

        # ── 2. Handle confirmation / rejection ────────────────────────────────
        if user_act == UserAct.CONFIRM and state.confirmation_status == ConfirmationStatus.PENDING:
            new_confirmation = ConfirmationStatus.CONFIRMED
        elif user_act == UserAct.REJECT:
            if state.pending_action is not None:
                new_confirmation = ConfirmationStatus.REJECTED
                new_pending = None
            # Rejection is not an intent switch; keep current intent

        # ── 3. Handle intent switch ───────────────────────────────────────────
        new_intent = understanding.primary_intent
        if user_act == UserAct.SWITCH and new_intent != state.active_intent:
            # Intent switching: keep only cross-reuse slots, clear pending action
            new_active = new_intent
            cross_slots = {
                k: v for k, v in new_slots.items()
                if k in _CROSS_INTENT_SLOTS
            }
            # Merge any new slots from the understanding
            cross_slots.update(understanding.extracted_slots)
            new_slots = cross_slots
            new_pending = None
            new_confirmation = ConfirmationStatus.NOT_REQUIRED
            new_queued = []
        elif new_active is None:
            # The first understood intent starts the goal. Later turns keep the
            # active goal unless the NLU explicitly marks a switch.
            new_active = new_intent

        # ── 4. Merge extracted slots ───────────────────────────────────────────
        for slot_name, slot_value in understanding.extracted_slots.items():
            if slot_value is None or slot_value == "":
                continue
            if slot_name in new_slots and new_slots[slot_name] != slot_value:
                # Only overwrite an existing slot value if the understanding
                # explicitly marks it as corrected.  Silent overwrites are
                # rejected to prevent accidental loss of user-provided values.
                if slot_name in understanding.corrected_slots:
                    logger.debug(f"Slot corrected: {slot_name} {new_slots[slot_name]} -> {slot_value}")
                    new_slots[slot_name] = slot_value
                else:
                    logger.debug(f"Slot overwrite suppressed: {slot_name} keeps {new_slots[slot_name]} (new={slot_value})")
            elif slot_name not in new_slots:
                new_slots[slot_name] = slot_value

        # ── 5. Merge observations (tool results) first ──────────────────────────
        if observations:
            for obs in observations:
                if obs.success and isinstance(obs.data, dict):
                    for k, v in obs.data.items():
                        # Observations fill slots but never override user-provided values
                        if k not in new_slots or not new_slots[k]:
                            new_slots[k] = v

        # ── 6. Compute required & missing slots after all slot sources ──────────
        if new_active and new_active in INTENT_SCHEMAS:
            schema = INTENT_SCHEMAS[new_active]
            new_required = list(schema.required_slots)
            new_missing = get_missing_slots(new_active, new_slots)
            if schema.allowed_agents:
                new_last_agent = schema.allowed_agents[0]

        # ── 7. Build new state ─────────────────────────────────────────────────
        return DialogueState(
            active_intent=new_active,
            slots=new_slots,
            required_slots=new_required,
            missing_slots=new_missing,
            pending_action=new_pending,
            confirmation_status=new_confirmation,
            completed_goals=new_completed,
            queued_goals=new_queued,
            last_agent=new_last_agent,
            state_version=state.state_version + 1,
        )

    @staticmethod
    def mark_goal_completed(state: DialogueState, goal: str) -> DialogueState:
        """Return a new state with ``goal`` added to completed_goals."""
        completed = list(state.completed_goals)
        if goal not in completed:
            completed.append(goal)
        return DialogueState(
            active_intent=state.active_intent,
            slots=dict(state.slots),
            required_slots=list(state.required_slots),
            missing_slots=list(state.missing_slots),
            pending_action=state.pending_action,
            confirmation_status=state.confirmation_status,
            completed_goals=completed,
            queued_goals=list(state.queued_goals),
            last_agent=state.last_agent,
            state_version=state.state_version + 1,
        )

    @staticmethod
    def set_pending_action(
        state: DialogueState,
        action: PendingAction,
    ) -> DialogueState:
        """Return a new state with a pending action and PENDING confirmation."""
        return DialogueState(
            active_intent=state.active_intent,
            slots=dict(state.slots),
            required_slots=list(state.required_slots),
            missing_slots=list(state.missing_slots),
            pending_action=action,
            confirmation_status=ConfirmationStatus.PENDING,
            completed_goals=list(state.completed_goals),
            queued_goals=list(state.queued_goals),
            last_agent=state.last_agent,
            state_version=state.state_version + 1,
        )
