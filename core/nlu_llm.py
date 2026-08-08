"""Structured LLM NLU: single-call intent + slot extraction with degradation.

When the deterministic fast-track cannot fully resolve the user's input,
this module calls the LLM once to produce a structured UnderstandingResult.
Invalid or unparseable LLM output is degraded to a safe fallback rather
than propagating errors.
"""
import json
import logging
from typing import Any, Dict, List, Optional

from core.agent_models import INTENT_SCHEMAS, UnderstandingResult, UserAct
from core.nlu_fast_track import fast_track_extract

logger = logging.getLogger(__name__)

# ── LLM prompt template ───────────────────────────────────────────────────────

_SYSTEM_PROMPT = """你是客服对话理解专家。根据用户消息和已有状态，提取意图和槽位。

返回 JSON，格式：
{{
  "intent": "<意图名>",
  "intents": ["<按优先级排列的全部意图>"],
  "confidence": <0-1>,
  "slots": {{"order_id": "...", ...}},
  "user_act": "<inform|confirm|reject|switch|ask>",
  "corrected_slots": ["<被用户纠正的槽位名>"]
}}

可用意图: {intents}
各意图必填槽位: {slot_info}

规则：
1. 如果用户纠正了之前提供的值，把该槽位名放入 corrected_slots。
2. 如果用户切换了目标意图，user_act 设为 switch。
3. 如果一句话包含多个独立目标，全部写入 intents，intent 为首要目标。
4. 仅输出 JSON，不要附加其他文字。"""

_FEW_SHOT = """
示例:
用户: "我要退款，订单号 ORD-1001"
→ {"intent":"refund_request","confidence":0.95,"slots":{"order_id":"ORD-1001"},"user_act":"inform","corrected_slots":[]}

用户: "不对，订单号是 ORD-1002"
→ {"intent":"refund_request","confidence":0.9,"slots":{"order_id":"ORD-1002"},"user_act":"inform","corrected_slots":["order_id"]}

用户: "算了不退了，我想查物流"
→ {"intent":"logistics_query","confidence":0.9,"slots":{},"user_act":"switch","corrected_slots":[]}
"""


def _build_prompt(
    text: str,
    current_state: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Build the full LLM prompt including system instructions and few-shot."""
    intent_names = ", ".join(sorted(INTENT_SCHEMAS.keys()))
    slot_info_parts = []
    for name, schema in INTENT_SCHEMAS.items():
        if schema.required_slots:
            slot_info_parts.append(f"{name}: {', '.join(schema.required_slots)}")
    slot_info = "; ".join(slot_info_parts) if slot_info_parts else "none"

    system = _SYSTEM_PROMPT.format(intents=intent_names, slot_info=slot_info)

    parts = [system, _FEW_SHOT]
    if history:
        recent = []
        remaining = 1600
        for item in reversed(history[-5:]):
            role = str(item.get("role", "user"))
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            content = content[:remaining]
            recent.append(f"{role}: {content}")
            remaining -= len(content)
            if remaining <= 0:
                break
        if recent:
            parts.append("最近对话:\n" + "\n".join(reversed(recent)))
    if current_state:
        parts.append(f"当前状态: {json.dumps(current_state, ensure_ascii=False)}")
    parts.append(f'用户消息: "{text}"')
    parts.append("→")

    return "\n".join(parts)


def _parse_llm_json(raw: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from LLM output text."""
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    s = text.find("{")
    e = text.rfind("}") + 1
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(text[s:e])
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
) -> UnderstandingResult:
    """Call LLM once for structured intent + slot extraction.

    Args:
        text: User message.
        llm_call_fn: Async callable that takes a prompt string and returns
                     raw LLM text output. This abstraction allows testing
                     without a real LLM client.
        current_state: Optional serialized DialogueState for context.

    Returns:
        UnderstandingResult on success, or a degraded fallback on failure.
    """
    prompt = _build_prompt(text, current_state, history)

    try:
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
