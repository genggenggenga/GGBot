"""Structured LLM NLU: single-call intent + slot extraction with degradation.

When the deterministic fast-track cannot fully resolve the user's input,
this module calls the LLM once to produce a structured UnderstandingResult.
Invalid or unparseable LLM output is degraded to a safe fallback rather
than propagating errors.
"""
import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from core.agent_models import INTENT_SCHEMAS, UnderstandingResult, UserAct
from core.nlu_fast_track import fast_track_extract
from core.prompts.nlu import build_prompt as build_nlu_prompt
from core.prompts.types import PromptSpec

logger = logging.getLogger(__name__)


class _NLUOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str
    intents: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    slots: Dict[str, Any] = Field(default_factory=dict)
    corrected_slots: List[str] = Field(default_factory=list)
    user_act: str = "inform"


def _build_prompt(
    text: str,
    current_state: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Build a combined prompt for diagnostics and backwards compatibility."""
    return _build_prompt_spec(text, current_state, history).combined()


def _build_prompt_spec(
    text: str,
    current_state: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> PromptSpec:
    """Build the provider-neutral system/user prompt pair."""
    slot_requirements = {
        name: list(schema.required_slots)
        for name, schema in INTENT_SCHEMAS.items()
        if schema.required_slots
    }
    return build_nlu_prompt(
        text,
        intent_names=sorted(INTENT_SCHEMAS),
        slot_requirements=slot_requirements,
        current_state=current_state,
        history=history,
    )


def _parse_llm_json(raw: str) -> Optional[Dict[str, Any]]:
    """Parse a strict JSON object from a test or legacy adapter."""
    text = raw.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _validate_llm_output(data: Dict[str, Any]) -> Optional[UnderstandingResult]:
    """Validate parsed LLM output and build UnderstandingResult, or None."""
    intent = data.get("intent", "")
    if not isinstance(intent, str) or not intent.strip():
        return None

    confidence = data.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    confidence = max(0.0, min(1.0, float(confidence)))

    slots = data.get("slots", {})
    if not isinstance(slots, dict):
        slots = {}

    act_str = data.get("user_act", "inform")
    try:
        user_act = UserAct(act_str)
    except ValueError:
        user_act = UserAct.INFORM

    corrected = data.get("corrected_slots", [])
    if not isinstance(corrected, list):
        corrected = []

    # Ensure intent is known; fallback to "other"
    primary_intent = intent if intent in INTENT_SCHEMAS else "other"
    if primary_intent != intent:
        confidence = min(confidence, 0.5)

    raw_intents = data.get("intents", [])
    intents = [
        str(candidate)
        for candidate in raw_intents
        if str(candidate) in INTENT_SCHEMAS
    ] if isinstance(raw_intents, list) else []
    if primary_intent in intents:
        intents.remove(primary_intent)
    intents.insert(0, primary_intent)

    try:
        return UnderstandingResult(
            intents=intents,
            primary_intent=primary_intent,
            confidence=confidence,
            extracted_slots=slots,
            corrected_slots=[str(c) for c in corrected],
            user_act=user_act,
            route_to=INTENT_SCHEMAS[primary_intent].allowed_agents[0]
                if primary_intent in INTENT_SCHEMAS else None,
        )
    except Exception:
        return None


def make_fallback_understanding(
    text: str,
    current_state: Optional[Dict[str, Any]] = None,
) -> UnderstandingResult:
    """Produce a safe OTHER/clarify fallback when LLM output is unusable."""
    ft = fast_track_extract(
        text,
        confirmation_pending=(
            (current_state or {}).get("confirmation_status") == "pending"
        ),
    )
    slots = dict(ft.slots) if ft.hit else {}
    user_act = ft.user_act or UserAct.INFORM
    primary_intent = ft.intent if ft.intent and ft.intent in INTENT_SCHEMAS else "other"
    if primary_intent == "other":
        if ft.order_id:
            primary_intent = "order_query"
        elif ft.tracking_no:
            primary_intent = "logistics_query"
    confidence = 0.3 if primary_intent == "other" else 0.6

    return UnderstandingResult(
        intents=[primary_intent],
        primary_intent=primary_intent,
        confidence=confidence,
        extracted_slots=slots,
        corrected_slots=list(ft.corrected_slots),
        user_act=user_act,
        route_to=INTENT_SCHEMAS[primary_intent].allowed_agents[0]
            if primary_intent in INTENT_SCHEMAS else None,
    )


def _normalize_user_act(
    result: UnderstandingResult,
    current_state: Optional[Dict[str, Any]],
) -> UnderstandingResult:
    """Apply confirmation-state and correction precedence to LLM output."""
    user_act = result.user_act
    if result.corrected_slots and user_act in {
        UserAct.CONFIRM,
        UserAct.REJECT,
    }:
        user_act = UserAct.INFORM
    elif (
        user_act in {UserAct.CONFIRM, UserAct.REJECT}
        and (current_state or {}).get("confirmation_status") != "pending"
    ):
        user_act = UserAct.INFORM
    if user_act == result.user_act:
        return result
    return result.model_copy(update={"user_act": user_act})


async def understand_with_llm(
    text: str,
    llm_call_fn: Any,
    current_state: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    structured_client: Any = None,
) -> UnderstandingResult:
    """Call LLM once for structured intent + slot extraction.

    Args:
        text: User message.
        llm_call_fn: Async callable that takes a PromptSpec and returns raw LLM
                     text output. This abstraction allows testing without a
                     real LLM client.
        current_state: Optional serialized DialogueState for context.

    Returns:
        UnderstandingResult on success, or a degraded fallback on failure.
    """
    prompt = _build_prompt_spec(text, current_state, history)

    try:
        if structured_client is not None:
            output = await structured_client.generate(
                prompt,
                _NLUOutput,
                tool_name="submit_understanding",
                max_tokens=512,
                temperature=0.1,
            )
            data = output.model_dump()
        else:
            raw = await llm_call_fn(prompt)
            if not isinstance(raw, str) or not raw.strip():
                logger.warning("LLM returned empty output, degrading")
                return make_fallback_understanding(text, current_state)

            data = _parse_llm_json(raw)
            if data is None:
                logger.warning("LLM output is not valid JSON, degrading")
                return make_fallback_understanding(text, current_state)

        result = _validate_llm_output(data)
        if result is None:
            logger.warning("LLM output failed validation, degrading")
            return make_fallback_understanding(text, current_state)

        return _normalize_user_act(result, current_state)

    except Exception as ex:
        logger.warning(f"LLM structured call failed: {ex}, degrading")
        return make_fallback_understanding(text, current_state)
