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
    r"(?:不(?:太)?可以|不行|不能|不办了|取消|不要|拒绝|算了|不对|no|cancel|reject|deny|不要了)",
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
    (re.compile(r"投诉|complain", re.IGNORECASE), "complaint"),
    (re.compile(r"转人工|人工客服|人工处理|escalat", re.IGNORECASE), "escalation"),
    (re.compile(r"退款政策|退款规则|退款条件|refund.?polic", re.IGNORECASE), "refund_policy"),
    (re.compile(r"退款|refund", re.IGNORECASE), "refund_request"),
    (re.compile(r"退货|return", re.IGNORECASE), "return_request"),
    (re.compile(r"取消订单|cancel", re.IGNORECASE), "cancel_order"),
    (
        re.compile(
            r"(?:配送|快递).*(?:一般|通常|几天|多久|费用|收费|政策|说明)",
            re.IGNORECASE,
        ),
        "query",
    ),
    (re.compile(r"物流|快递|配送|发货|track", re.IGNORECASE), "logistics_query"),
    (re.compile(r"订单|order", re.IGNORECASE), "order_query"),
]

_EXPLICIT_INTENT_PATTERNS: Dict[str, re.Pattern] = {
    "complaint": re.compile(r"投诉|complain", re.IGNORECASE),
    "escalation": re.compile(r"转人工|人工客服|人工处理|escalat", re.IGNORECASE),
    "refund_policy": re.compile(r"退款政策|退款规则|退款条件|refund.?polic", re.IGNORECASE),
    "refund_request": re.compile(
        r"(?:我要|申请|办理|帮我|需要|发起).{0,8}(?:退款|refund)",
        re.IGNORECASE,
    ),
    "return_request": re.compile(
        r"(?:我要|申请|办理|帮我|需要|发起).{0,8}(?:退货|return)",
        re.IGNORECASE,
    ),
    "cancel_order": re.compile(r"取消订单|cancel\s+(?:my\s+)?order", re.IGNORECASE),
    "logistics_query": re.compile(
        r"(?:查|查询|看看|看下|追踪|跟踪).{0,8}(?:物流|快递|配送|track)",
        re.IGNORECASE,
    ),
    "order_query": re.compile(
        r"(?:查|查询|看看|看下).{0,8}(?:订单|order)",
        re.IGNORECASE,
    ),
}


def extract_order_id(text: str) -> Optional[str]:
    """Extract the first matching order ID from text."""
    for pat in _ORDER_ID_PATTERNS:
        m = pat.search(text)
        if m:
            value = m.group(1).upper()
            if re.fullmatch(r"(?:SF|YT|JD|ZTO|STO|YD|EMS)\d{8,15}", value):
                continue
            return value
    return None


def extract_tracking_no(text: str) -> Optional[str]:
    """Extract the first matching tracking number from text."""
    for pat in _TRACKING_NO_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    return None


def detect_user_act(
    text: str,
    *,
    confirmation_pending: bool = False,
) -> Optional[UserAct]:
    """Detect confirmation or rejection from user text, or None.

    When intent keywords are present (e.g. "好的退款"), the confirm/reject
    signal is suppressed so the intent takes priority.  A pure confirm/reject
    word without any intent keyword is treated as the user's primary signal.
    """
    intent = detect_intent_from_keywords(text)
    if intent is not None and _SWITCH_WORDS.search(text):
        return UserAct.SWITCH
    if not confirmation_pending:
        return None
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
    intents = detect_intents_from_keywords(text)
    return intents[0] if intents else None


def detect_intents_from_keywords(text: str) -> List[str]:
    """Return every distinct intent matched by deterministic patterns."""
    intents = []
    for pat, intent in _INTENT_KEYWORDS:
        if pat.search(text) and intent not in intents:
            intents.append(intent)
    if "refund_policy" in intents and "refund_request" in intents:
        intents.remove("refund_request")
    if "cancel_order" in intents and "order_query" in intents:
        intents.remove("order_query")
    if "complaint" in intents and "escalation" in intents:
        intents.remove("escalation")
    if "query" in intents and "logistics_query" in intents:
        intents.remove("logistics_query")
    return intents


class FastTrackResult:
    """Aggregate fast-track extraction results."""

    __slots__ = (
        "order_id",
        "tracking_no",
        "user_act",
        "intent",
        "intents",
        "slots",
        "corrected_slots",
        "explicit_intent",
        "hit",
    )

    def __init__(
        self,
        order_id: Optional[str] = None,
        tracking_no: Optional[str] = None,
        user_act: Optional[UserAct] = None,
        intent: Optional[str] = None,
        intents: Optional[List[str]] = None,
        corrected_slots: Optional[List[str]] = None,
        explicit_intent: bool = False,
    ):
        self.order_id = order_id
        self.tracking_no = tracking_no
        self.user_act = user_act
        self.intent = intent
        self.intents = intents or ([intent] if intent else [])
        self.corrected_slots = corrected_slots or []
        self.explicit_intent = explicit_intent

        self.slots: Dict[str, str] = {}
        if order_id:
            self.slots["order_id"] = order_id
        if tracking_no:
            self.slots["tracking_no"] = tracking_no

        self.hit = bool(
            self.slots
            or self.user_act is not None
            or self.intents
        )


def fast_track_extract(
    text: str,
    *,
    confirmation_pending: bool = False,
) -> FastTrackResult:
    """Run all deterministic extractors on the input text.

    Returns a FastTrackResult with any matched fields. The ``hit`` flag
    indicates whether at least one rule fired.
    """
    order_id = extract_order_id(text)
    tracking_no = extract_tracking_no(text)
    user_act = detect_user_act(
        text,
        confirmation_pending=confirmation_pending,
    )
    intents = detect_intents_from_keywords(text)
    intent = intents[0] if intents else None
    explicit_intent = bool(
        intent
        and (
            (pattern := _EXPLICIT_INTENT_PATTERNS.get(intent)) is not None
            and pattern.search(text)
        )
    )
    corrected_slots: List[str] = []
    if _CORRECTION_WORDS.search(text):
        if order_id:
            corrected_slots.append("order_id")
        if tracking_no:
            corrected_slots.append("tracking_no")
    if corrected_slots:
        user_act = UserAct.INFORM

    return FastTrackResult(
        order_id=order_id,
        tracking_no=tracking_no,
        user_act=user_act,
        intent=intent,
        intents=intents,
        corrected_slots=corrected_slots,
        explicit_intent=explicit_intent,
    )


def build_understanding_from_fast_track(
    ft: FastTrackResult,
    text: str,
) -> Optional[UnderstandingResult]:
    """Convert a fast-track result into an UnderstandingResult.

    Returns None if the fast-track did not produce enough signal to
    fully determine the user's intent and slots.
    """
    del text
    if not ft.hit:
        return None

    primary_intent = ft.intent or "other"
    # If we detected slots but no intent, try inferring from slots
    if ft.intent is None and ft.order_id:
        primary_intent = "order_query"
    elif ft.intent is None and ft.tracking_no:
        primary_intent = "logistics_query"

    schema = INTENT_SCHEMAS.get(primary_intent)
    slots_complete = False
    if schema and schema.required_slots:
        present = [bool(ft.slots.get(slot)) for slot in schema.required_slots]
        slots_complete = (
            any(present)
            if schema.slot_mode.value == "any"
            else all(present)
        )

    if ft.user_act in {UserAct.CONFIRM, UserAct.REJECT}:
        confidence = 1.0
    elif ft.user_act == UserAct.SWITCH and ft.intent:
        confidence = 0.95
    elif ft.intent and slots_complete:
        confidence = 1.0
    elif ft.explicit_intent or (ft.intent is None and ft.slots):
        confidence = 0.9
    else:
        confidence = 0.6

    # Validate intent is known
    if primary_intent not in INTENT_SCHEMAS:
        primary_intent = "other"
        confidence = 0.7

    intents = [
        intent
        for intent in ft.intents
        if intent in INTENT_SCHEMAS
    ] or [primary_intent]
    if primary_intent not in intents:
        intents.insert(0, primary_intent)

    return UnderstandingResult(
        intents=intents,
        primary_intent=primary_intent,
        confidence=confidence,
        extracted_slots=dict(ft.slots),
        corrected_slots=list(ft.corrected_slots),
        user_act=ft.user_act or UserAct.INFORM,
        route_to=INTENT_SCHEMAS[primary_intent].allowed_agents[0]
            if primary_intent in INTENT_SCHEMAS else None,
    )
