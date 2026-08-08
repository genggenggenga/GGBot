"""Agent runtime shared models and intent contracts."""
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class UserAct(str, Enum):
    INFORM = "inform"
    CONFIRM = "confirm"
    REJECT = "reject"
    SWITCH = "switch"
    ASK = "ask"


class ConfirmationStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class SlotMode(str, Enum):
    ALL = "all"
    ANY = "any"


class ExecutionState(str, Enum):
    UNDERSTANDING = "understanding"
    CLARIFYING = "clarifying"
    ROUTING = "routing"
    RETRIEVING = "retrieving"
    ACTING = "acting"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    RESPONDING = "responding"
    COMPLETED = "completed"
    FAILED = "failed"


class PendingAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(default_factory=lambda: str(uuid4()))
    tool_name: str = Field(min_length=1)
    arguments: Dict[str, Any] = Field(default_factory=dict)
    requires_confirmation: bool = True


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    name: str = Field(min_length=1)
    success: bool
    data: Any = None
    error: Optional[str] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    @model_validator(mode="after")
    def validate_error(self) -> "Observation":
        if not self.success and not self.error:
            raise ValueError("failed observation requires an error")
        return self


class UnderstandingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intents: List[str] = Field(min_length=1)
    primary_intent: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    extracted_slots: Dict[str, Any] = Field(default_factory=dict)
    corrected_slots: List[str] = Field(default_factory=list)
    user_act: UserAct = UserAct.INFORM
    route_to: Optional[str] = None

    @field_validator("intents")
    @classmethod
    def validate_intents(cls, intents: List[str]) -> List[str]:
        if any(not intent.strip() for intent in intents):
            raise ValueError("intents cannot contain empty values")
        return list(dict.fromkeys(intents))

    @model_validator(mode="after")
    def validate_primary_intent(self) -> "UnderstandingResult":
        if self.primary_intent not in self.intents:
            raise ValueError("primary_intent must be included in intents")
        return self


class DialogueState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_intent: Optional[str] = None
    slots: Dict[str, Any] = Field(default_factory=dict)
    required_slots: List[str] = Field(default_factory=list)
    missing_slots: List[str] = Field(default_factory=list)
    pending_action: Optional[PendingAction] = None
    confirmation_status: ConfirmationStatus = ConfirmationStatus.NOT_REQUIRED
    completed_goals: List[str] = Field(default_factory=list)
    queued_goals: List[str] = Field(default_factory=list)
    last_agent: Optional[str] = None
    state_version: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_missing_slots(self) -> "DialogueState":
        unknown_slots = set(self.missing_slots) - set(self.required_slots)
        if unknown_slots:
            names = ", ".join(sorted(unknown_slots))
            raise ValueError(f"missing_slots must be required slots: {names}")
        return self


class Transition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    next_state: ExecutionState
    dialogue_updates: Dict[str, Any] = Field(default_factory=dict)
    observations: List[Observation] = Field(default_factory=list)
    response: Optional[str] = None
    reason: str = Field(min_length=1)


class HandoffPackage(BaseModel):
    """Structured context passed to a human when automated execution stops."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1)
    active_intent: Optional[str] = None
    slots: Dict[str, Any] = Field(default_factory=dict)
    observations: List[Observation] = Field(default_factory=list)
    failed_state: ExecutionState
    suggested_next_step: str = "人工客服继续处理"


class TurnContext(BaseModel):
    """Mutable runtime snapshot for one bounded engine execution."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1)
    conv_id: str = Field(min_length=1)
    dialogue_state: DialogueState = Field(default_factory=DialogueState)
    execution_state: ExecutionState = ExecutionState.UNDERSTANDING
    state_history: List[ExecutionState] = Field(
        default_factory=lambda: [ExecutionState.UNDERSTANDING],
    )
    observations: List[Observation] = Field(default_factory=list)
    step_count: int = Field(default=0, ge=0)
    response: Optional[str] = None
    handoff: Optional[HandoffPackage] = None


class IntentSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: str = Field(min_length=1)
    required_slots: tuple[str, ...] = ()
    slot_mode: SlotMode = SlotMode.ALL
    allowed_agents: tuple[str, ...] = Field(min_length=1)
    completion_condition: str = Field(min_length=1)


def _intent(
    name: str,
    agents: tuple[str, ...],
    completion_condition: str,
    required_slots: tuple[str, ...] = (),
    slot_mode: SlotMode = SlotMode.ALL,
) -> IntentSchema:
    return IntentSchema(
        intent=name,
        required_slots=required_slots,
        slot_mode=slot_mode,
        allowed_agents=agents,
        completion_condition=completion_condition,
    )


_KNOWLEDGE_AGENT = ("knowledge",)
_AFTER_SALES_AGENT = ("after_sales",)

INTENT_SCHEMAS: Mapping[str, IntentSchema] = MappingProxyType({
    "query": _intent("query", _KNOWLEDGE_AGENT, "answer_grounded_in_knowledge"),
    "complaint": _intent("complaint", _AFTER_SALES_AGENT, "handoff_created"),
    "request": _intent("request", _AFTER_SALES_AGENT, "requested_action_completed"),
    "greeting": _intent("greeting", _KNOWLEDGE_AGENT, "response_generated"),
    "escalation": _intent("escalation", _AFTER_SALES_AGENT, "handoff_created"),
    "technical": _intent("technical", _KNOWLEDGE_AGENT, "answer_grounded_in_knowledge"),
    "billing": _intent("billing", ("knowledge", "after_sales"), "billing_goal_resolved"),
    "account": _intent("account", _KNOWLEDGE_AGENT, "answer_grounded_in_knowledge"),
    "feedback": _intent("feedback", _KNOWLEDGE_AGENT, "response_generated"),
    "other": _intent("other", _KNOWLEDGE_AGENT, "response_or_clarification_generated"),
    "order_query": _intent(
        "order_query",
        ("order",),
        "order_fact_returned",
        ("order_id",),
    ),
    "logistics_query": _intent(
        "logistics_query",
        ("logistics",),
        "logistics_fact_returned",
        ("order_id", "tracking_no"),
        SlotMode.ANY,
    ),
    "refund_policy": _intent(
        "refund_policy",
        _KNOWLEDGE_AGENT,
        "answer_grounded_in_knowledge",
    ),
    "refund_request": _intent(
        "refund_request",
        _AFTER_SALES_AGENT,
        "refund_created_or_handoff_created",
        ("order_id",),
    ),
    "return_request": _intent(
        "return_request",
        _AFTER_SALES_AGENT,
        "return_created_or_handoff_created",
        ("order_id",),
    ),
    "cancel_order": _intent(
        "cancel_order",
        _AFTER_SALES_AGENT,
        "order_cancelled_or_handoff_created",
        ("order_id",),
    ),
})


def get_intent_schema(intent: str) -> IntentSchema:
    """Return the contract for a supported intent."""
    try:
        return INTENT_SCHEMAS[intent]
    except KeyError as ex:
        raise ValueError(f"unsupported intent: {intent}") from ex


def get_missing_slots(intent: str, slots: Mapping[str, Any]) -> List[str]:
    """Return unsatisfied slot requirements for an intent."""
    schema = get_intent_schema(intent)
    if schema.slot_mode == SlotMode.ANY:
        return (
            []
            if any(slots.get(slot) for slot in schema.required_slots)
            else list(schema.required_slots)
        )
    return [
        slot for slot in schema.required_slots
        if not slots.get(slot)
    ]
