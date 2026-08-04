"""Deterministic fast-track channel for structured NLU.

Extracts order IDs, tracking numbers, and confirmation/rejection keywords
using pure regex rules. When a fast-track hits, the caller can skip the
LLM call for the matched fields entirely.
"""
import re
from typing import Dict, List, Optional, Tuple

from core.agent_models import UnderstandingResult, UserAct, INTENT_SCHEMAS

# ── Regex patterns ────────────────────────────────────────────────────────────

# Order IDs: ORD-XXXX, ORDER-XXXX, or Chinese prefix "订单号" / "订单" + digits/alphanum
_ORDER_ID_PATTERNS = [
    re.compile(r"(?:订单号?[:：\s]*)?([A-Za-z]*ORD(?:ER)?[-_]?[\dA-Za-z]{4,})", re.IGNORECASE),
    re.compile(r"订单号?[:：\s]+(\d{6,})", re.IGNORECASE),
    re.compile(r"\b([A-Z]{2,4}\d{8,12})\b"),  # generic alphanumeric codes
]

# Tracking numbers: SF1234567890, YT123456789, JD1234567890, etc.
_TRACKING_NO_PATTERNS = [
    re.compile(r"(?:物流|快递|运单号?[:：\s]*)?((?:SF|YT|JD|ZTO|STO|YD|EMS)\d{8,15})", re.IGNORECASE),
    re.compile(r"运单号?[:：\s]+(\d{8,20})", re.IGNORECASE),
]

# Confirm words
_CONFIRM_WORDS = re.compile(
    r"(?:确认|是的|对|没问题|可以|好的|确认提交|yes|ok|sure|confirm)",
    re.IGNORECASE,
)

# Reject words
_REJECT_WORDS = re.compile(
    r"(?:不(?:太)?可以|不行|不能|取消|不要|拒绝|算了|不对|no|cancel|reject|deny|不要了)",
    re.IGNORECASE,
)

_SWITCH_WORDS = re.compile(
    r"(?:算了|不退了|改(?:成|为|查)|换(?:成|为)|转(?:成|为))",
    re.IGNORECASE,
)

_CORRECTION_WORDS = re.compile(
    r"(?:不对|不是|更正|改成|应该是|说错了|correct)",
    re.IGNORECASE,
)

# Intent keyword map (word -> intent)
_INTENT_KEYWORDS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"退款政策|退款规则|退款条件|refund.?polic", re.IGNORECASE), "refund_policy"),
    (re.compile(r"退款|refund", re.IGNORECASE), "refund_request"),
    (re.compile(r"退货|return", re.IGNORECASE), "return_request"),
    (re.compile(r"取消订单|cancel", re.IGNORECASE), "cancel_order"),
    (re.compile(r"物流|快递|配送|发货|track", re.IGNORECASE), "logistics_query"),
    (re.compile(r"订单|order", re.IGNORECASE), "order_query"),
]


def extract_order_id(text: str) -> Optional[str]:
    """Extract the first matching order ID from text."""
    for pat in _ORDER_ID_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    return None


def extract_tracking_no(text: str) -> Optional[str]:
    """Extract the first matching tracking number from text."""
    for pat in _TRACKING_NO_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    return None


def detect_user_act(text: str) -> Optional[UserAct]:
    """Detect confirmation or rejection from user text, or None.

    When intent keywords are present (e.g. "好的退款"), the confirm/reject
    signal is suppressed so the intent takes priority.  A pure confirm/reject
    word without any intent keyword is treated as the user's primary signal.
    """
    intent = detect_intent_from_keywords(text)
    if intent is not None and _SWITCH_WORDS.search(text):
        return UserAct.SWITCH
    if _REJECT_WORDS.search(text):
        return UserAct.REJECT
    # If the text also carries an intent keyword, the user is expressing
    # that intent, not confirming a pending action.
    if intent is not None:
        return None
    if _CONFIRM_WORDS.search(text):
        return UserAct.CONFIRM
    return None


def detect_intent_from_keywords(text: str) -> Optional[str]:
    """Detect intent from keyword patterns, returning the first match."""
    for pat, intent in _INTENT_KEYWORDS:
        if pat.search(text):
            return intent
    return None


class FastTrackResult:
    """Aggregate fast-track extraction results."""

    __slots__ = (
        "order_id",
        "tracking_no",
        "user_act",
        "intent",
        "slots",
        "corrected_slots",
        "hit",
    )

    def __init__(
        self,
        order_id: Optional[str] = None,
        tracking_no: Optional[str] = None,
        user_act: Optional[UserAct] = None,
        intent: Optional[str] = None,
        corrected_slots: Optional[List[str]] = None,
    ):
        self.order_id = order_id
        self.tracking_no = tracking_no
        self.user_act = user_act
        self.intent = intent
        self.corrected_slots = corrected_slots or []

        self.slots: Dict[str, str] = {}
        if order_id:
            self.slots["order_id"] = order_id
        if tracking_no:
            self.slots["tracking_no"] = tracking_no

        self.hit = bool(self.slots or self.user_act is not None or self.intent is not None)


def fast_track_extract(text: str) -> FastTrackResult:
    """Run all deterministic extractors on the input text.

    Returns a FastTrackResult with any matched fields. The ``hit`` flag
    indicates whether at least one rule fired.
    """
    order_id = extract_order_id(text)
    tracking_no = extract_tracking_no(text)
    user_act = detect_user_act(text)
    intent = detect_intent_from_keywords(text)
    corrected_slots: List[str] = []
    if _CORRECTION_WORDS.search(text):
        if order_id:
            corrected_slots.append("order_id")
        if tracking_no:
            corrected_slots.append("tracking_no")

    return FastTrackResult(
        order_id=order_id,
        tracking_no=tracking_no,
        user_act=user_act,
        intent=intent,
        corrected_slots=corrected_slots,
    )


def build_understanding_from_fast_track(
    ft: FastTrackResult,
    text: str,
) -> Optional[UnderstandingResult]:
    """Convert a fast-track result into an UnderstandingResult.

    Returns None if the fast-track did not produce enough signal to
    fully determine the user's intent and slots.
    """
    if not ft.hit:
        return None

    primary_intent = ft.intent or "other"
    # If we detected slots but no intent, try inferring from slots
    if ft.intent is None and ft.order_id:
        primary_intent = "order_query"

    confidence = 1.0 if (ft.intent and ft.slots) else 0.9

    # Validate intent is known
    if primary_intent not in INTENT_SCHEMAS:
        primary_intent = "other"
        confidence = 0.7

    return UnderstandingResult(
        intents=[primary_intent],
        primary_intent=primary_intent,
        confidence=confidence,
        extracted_slots=dict(ft.slots),
        corrected_slots=list(ft.corrected_slots),
        user_act=ft.user_act or UserAct.INFORM,
        route_to=INTENT_SCHEMAS[primary_intent].allowed_agents[0]
            if primary_intent in INTENT_SCHEMAS else None,
    )
