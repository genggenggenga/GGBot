import pytest
from pydantic import ValidationError

from core.agent_models import (
    INTENT_SCHEMAS,
    ConfirmationStatus,
    DialogueState,
    ExecutionState,
    Observation,
    PendingAction,
    SlotMode,
    Transition,
    TurnEventType,
    UnderstandingResult,
    UserAct,
    get_intent_schema,
)


def test_dialogue_state_defaults_are_isolated():
    first = DialogueState()
    second = DialogueState()

    first.slots["order_id"] = "ORDER-1001"
    first.required_slots.append("order_id")
    first.completed_goals.append("order_query")

    assert second.slots == {}
    assert second.required_slots == []
    assert second.completed_goals == []
    assert second.state_version == 0
    assert second.confirmation_status == ConfirmationStatus.NOT_REQUIRED


def test_runtime_models_support_json_round_trip():
    understanding = UnderstandingResult(
        intents=["refund_request", "billing"],
        primary_intent="refund_request",
        confidence=0.92,
        extracted_slots={"order_id": "ORDER-1001"},
        corrected_slots=["order_id"],
        user_act=UserAct.INFORM,
        route_to="after_sales",
    )
    action = PendingAction(
        tool_name="create_refund",
        arguments={"order_id": "ORDER-1001"},
    )
    state = DialogueState(
        active_intent=understanding.primary_intent,
        slots=understanding.extracted_slots,
        required_slots=["order_id"],
        pending_action=action,
        confirmation_status=ConfirmationStatus.PENDING,
        last_agent="after_sales",
        state_version=3,
    )
    observation = Observation(
        source="tool",
        name="query_order",
        success=True,
        data={"status": "paid"},
    )
    transition = Transition(
        next_state=ExecutionState.AWAITING_CONFIRMATION,
        event=TurnEventType.WRITE_CONFIRMATION_REQUIRED,
        dialogue_updates=state.model_dump(mode="json"),
        observations=[observation],
        response="确认提交退款吗？",
        reason="refund requires confirmation",
    )

    restored = Transition.model_validate_json(transition.model_dump_json())

    assert restored == transition
    assert restored.observations[0].created_at.tzinfo is not None
    assert state.pending_action is not None
    assert state.pending_action.action_id


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_understanding_result_rejects_invalid_confidence(confidence):
    with pytest.raises(ValidationError):
        UnderstandingResult(
            intents=["refund_request"],
            primary_intent="refund_request",
            confidence=confidence,
        )


def test_understanding_result_rejects_unknown_primary_intent():
    with pytest.raises(
        ValidationError,
        match="primary_intent must be included in intents",
    ):
        UnderstandingResult(
            intents=["billing"],
            primary_intent="refund_request",
            confidence=0.8,
        )


def test_understanding_result_rejects_empty_candidate_intent():
    with pytest.raises(
        ValidationError,
        match="intents cannot contain empty values",
    ):
        UnderstandingResult(
            intents=["refund_request", ""],
            primary_intent="refund_request",
            confidence=0.8,
        )


def test_transition_rejects_invalid_execution_state():
    with pytest.raises(ValidationError):
        Transition(
            next_state="unknown",
            event=TurnEventType.RESPONSE_READY,
            reason="invalid state",
        )


def test_dialogue_state_rejects_negative_version():
    with pytest.raises(ValidationError):
        DialogueState(state_version=-1)


def test_dialogue_state_rejects_missing_slot_outside_requirements():
    with pytest.raises(
        ValidationError,
        match="missing_slots must be required slots",
    ):
        DialogueState(
            required_slots=["order_id"],
            missing_slots=["refund_reason"],
        )


def test_failed_observation_requires_error():
    with pytest.raises(
        ValidationError,
        match="failed observation requires an error",
    ):
        Observation(
            source="tool",
            name="query_order",
            success=False,
        )


def test_pending_action_ids_are_unique():
    first = PendingAction(tool_name="create_refund")
    second = PendingAction(tool_name="create_refund")

    assert first.action_id != second.action_id


@pytest.mark.parametrize(
    "intent, required_slots, allowed_agents, completion_condition",
    [
        (
            "order_query",
            ("order_id",),
            ("order",),
            "order_fact_returned",
        ),
        (
            "logistics_query",
            ("order_id", "tracking_no"),
            ("logistics",),
            "logistics_fact_returned",
        ),
        (
            "refund_request",
            ("order_id",),
            ("after_sales",),
            "refund_created_or_handoff_created",
        ),
        (
            "refund_policy",
            (),
            ("knowledge",),
            "answer_grounded_in_knowledge",
        ),
    ],
)
def test_intent_schema_declares_runtime_contract(
    intent,
    required_slots,
    allowed_agents,
    completion_condition,
):
    schema = get_intent_schema(intent)

    assert schema.required_slots == required_slots
    assert schema.allowed_agents == allowed_agents
    assert schema.completion_condition == completion_condition


def test_logistics_intent_accepts_any_supported_identifier():
    schema = get_intent_schema("logistics_query")

    assert schema.slot_mode == SlotMode.ANY


def test_intent_schema_covers_existing_intent_categories():
    existing_intents = {
        "query",
        "complaint",
        "request",
        "greeting",
        "escalation",
        "technical",
        "billing",
        "account",
        "feedback",
        "other",
    }

    assert existing_intents <= INTENT_SCHEMAS.keys()


def test_intent_schema_is_read_only():
    with pytest.raises(TypeError):
        INTENT_SCHEMAS["new_intent"] = get_intent_schema("query")

    with pytest.raises(ValidationError):
        get_intent_schema("query").intent = "changed"


def test_get_intent_schema_rejects_unsupported_intent():
    with pytest.raises(ValueError, match="unsupported intent: unknown"):
        get_intent_schema("unknown")
