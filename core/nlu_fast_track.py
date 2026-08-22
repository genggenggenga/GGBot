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
    r"(?:确认|是的|对|没问题|可以|好的|确认提交)",
)

# Reject words
_REJECT_WORDS = re.compile(
    r"(?:不(?:太)?可以|不行|不能|不办了|取消|不要|拒绝|算了|不对|不要了)",
)

_SWITCH_WORDS = re.compile(
    r"(?:算了|不退了|改(?:成|为|查)|换(?:成|为)|转(?:成|为))",
)

_CORRECTION_WORDS = re.compile(
    r"(?:不对|不是|更正|改成|应该是|说错了)",
)

# Chinese customer-service intent patterns. Each pattern requires either an
# explicit action or a concrete service question, avoiding analytics phrases
# such as "退款率报表" and "订单数据统计".
_COMPLAINT_PATTERN = (
    r"(?:我要|我想|想要|申请|发起|提出|需要|帮我).{0,6}投诉"
    r"|投诉(?:商家|客服|服务|平台|店铺|卖家|处理|问题)"
    r"|^\s*投诉\s*$"
)
_ESCALATION_PATTERN = r"(?:转|找|联系).{0,4}(?:人工|客服)|人工(?:客服|处理)"
# 咨询类退款意图：规则/条件/流程/时效，以及“怎么申请退款”“退款怎么申请”
# 等询问如何操作的问题。此类问题应走 knowledge agent，而不是 after_sales。
_REFUND_POLICY_PATTERN = (
    r"退款(?:政策|规则|条件|流程|步骤|方法|期限|时效|说明|要求|标准)"
    r"|(?:怎么|如何|怎样|咋)(?!.{0,6}(?:还没|没有)).{0,10}退款"
    r"|(?:申请|办理|发起|提交)退款(?:的)?"
    r"(?:流程|步骤|方法|条件|规则|要求|材料|凭证|需要什么|怎么|如何|怎样)"
    r"|退款(?:怎么|如何|怎样)(?:申请|办理|发起|提交|退|操作|弄|走)?"
)
_REFUND_REQUEST_PATTERN = (
    r"(?:我要|我想|想要|申请|办理|帮我|需要|发起|提交|请求|要求|请)"
    r".{0,20}退款"
    r"|退款(?=.{0,16}(?:订单(?:号|编号)?|[A-Za-z]*ORD(?:ER)?[-_]?"
    r"[\dA-Za-z]{4,}))"
    r"|退款(?:申请|进度|状态|到账|失败|金额|结果|记录|处理|审核|多久|什么时候"
    r"|何时|怎么办|怎么|能否|可以|是否)"
    r"|(?:为什么|怎么|何时|什么时候|多久|还没|没有).{0,8}退款"
    r"|^\s*退款\s*$"
)
_RETURN_REQUEST_PATTERN = (
    r"(?:我要|我想|想要|申请|办理|帮我|需要|发起|提交|请求|要求|请)"
    r".{0,20}退货"
    r"|退货(?=.{0,16}(?:订单(?:号|编号)?|[A-Za-z]*ORD(?:ER)?[-_]?"
    r"[\dA-Za-z]{4,}))"
    r"|退货(?:申请|进度|状态|流程|地址|方式|失败|结果|处理|怎么办|怎么|能否"
    r"|可以|是否)"
    r"|^\s*退货\s*$"
)
_CANCEL_ORDER_PATTERN = (
    r"(?:取消|撤销|不要).{0,4}(?:这个|该|我的)?订单"
    r"|(?:这个|该|我的)?订单.{0,4}(?:取消|撤销|不要了)"
)

# 咨询“如何操作”类问法：怎么申请退货/取消订单/修改地址等，归入 query
# 走 knowledge agent，避免被误判为执行请求进入 after_sales。
_PROCEDURE_QUERY_PATTERN = (
    r"(?:怎么|如何|怎样|咋)(?!.{0,6}(?:还没|没有)).{0,10}"
    r"(?:申请|办理|发起|提交|操作|弄|退)?(?:退货|退款申请)"
    r"|(?:怎么|如何|怎样|咋).{0,10}取消(?:这个|该|我的)?订单"
    r"|(?:怎么|如何|怎样|咋).{0,10}(?:改|修改|变更)(?:收货)?地址"
)
# 咨询“能否查询他人订单/信息”类隐私问题，归入 query 走 knowledge agent
# 的 guardrail 知识，而不是被“查询…订单”误判为 order_query。
_PRIVACY_QUERY_PATTERN = (
    r"(?:客服|机器人|系统|你们).{0,8}(?:可以|能|能否|能不能|是否|允许|会)"
    r".{0,10}(?:查询|查看|看到|访问|透露).{0,8}(?:他人|别人|别的用户)?订单"
    r"|(?:查询|查看|看到).{0,6}(?:他人|别人|别的用户).{0,6}"
    r"(?:订单|信息|明细|轨迹)"
    r"|(?:他人|别人|别的用户).{0,4}(?:订单|信息|明细|轨迹)"
)
_DELIVERY_POLICY_PATTERN = (
    r"(?:配送|快递|物流).{0,12}(?:一般|通常|几天|多久|费用|收费|政策|说明|时效)"
    r"|(?:一般|通常|几天|多久|费用|收费|政策|说明|时效).{0,12}"
    r"(?:配送|快递|物流)"
)
_LOGISTICS_QUERY_PATTERN = (
    r"(?:查|查询|看看|看下|追踪|跟踪|帮我查|想知道).{0,12}"
    r"(?:物流|快递|配送|发货|运单)"
    r"|(?:物流|快递|配送|发货|运单).{0,10}"
    r"(?:状态|进度|到哪|到哪里|什么时候到|何时到|多久到|没更新|未更新|异常"
    r"|停滞|丢失|签收)"
    r"|(?:查|查询|看看|看下).{0,30}订单.{0,8}(?:和|及|与)?物流"
)
_ORDER_QUERY_PATTERN = (
    r"(?:查|查询|看看|看下|帮我查|想知道).{0,12}订单"
    r"|(?:我的|这个|该)?订单.{0,10}"
    r"(?:状态|进度|详情|信息|金额|支付|商品|发票|在哪|怎么样|怎么了|失败|异常)"
)

# Intent pattern map (pattern -> intent)
_INTENT_KEYWORDS: List[Tuple[re.Pattern, str]] = [
    (re.compile(_COMPLAINT_PATTERN), "complaint"),
    (re.compile(_ESCALATION_PATTERN), "escalation"),
    (re.compile(_REFUND_POLICY_PATTERN), "refund_policy"),
    (re.compile(_REFUND_REQUEST_PATTERN), "refund_request"),
    (re.compile(_PROCEDURE_QUERY_PATTERN), "query"),
    (re.compile(_RETURN_REQUEST_PATTERN), "return_request"),
    (re.compile(_CANCEL_ORDER_PATTERN), "cancel_order"),
    (re.compile(_DELIVERY_POLICY_PATTERN), "query"),
    (re.compile(_PRIVACY_QUERY_PATTERN), "query"),
    (re.compile(_LOGISTICS_QUERY_PATTERN), "logistics_query"),
    (re.compile(_ORDER_QUERY_PATTERN), "order_query"),
]

_EXPLICIT_INTENT_PATTERNS: Dict[str, re.Pattern] = {
    "complaint": re.compile(_COMPLAINT_PATTERN),
    "escalation": re.compile(_ESCALATION_PATTERN),
    "refund_policy": re.compile(_REFUND_POLICY_PATTERN),
    "query": re.compile(
        rf"(?:{_PROCEDURE_QUERY_PATTERN})|(?:{_PRIVACY_QUERY_PATTERN})",
    ),
    "refund_request": re.compile(
        r"(?:我要|我想|想要|申请|办理|帮我|需要|发起|提交|请求|要求|请)"
        r".{0,20}退款",
    ),
    "return_request": re.compile(
        r"(?:我要|我想|想要|申请|办理|帮我|需要|发起|提交|请求|要求|请)"
        r".{0,20}退货",
    ),
    "cancel_order": re.compile(_CANCEL_ORDER_PATTERN),
    "logistics_query": re.compile(
        r"(?:查|查询|看看|看下|追踪|跟踪|帮我查|想知道).{0,12}"
        r"(?:物流|快递|配送|发货|运单)",
    ),
    "order_query": re.compile(
        r"(?:查|查询|看看|看下|帮我查|想知道).{0,12}订单",
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
    # A concrete business intent takes priority over generic confirmation
    # words. For example, "取消订单" starts a new goal while "取消" rejects
    # the currently pending action.
    if intent is not None:
        return None
    if _REJECT_WORDS.search(text):
        return UserAct.REJECT
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
    if "query" in intents and "order_query" in intents:
        intents.remove("order_query")
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
