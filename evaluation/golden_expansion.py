"""Curated scenario matrix used to expand the Golden v1 candidate set.

The matrix is code so repeated policy and workflow variants remain auditable
and do not require copying hundreds of nearly identical JSON objects.
"""
from __future__ import annotations

from itertools import cycle
from typing import Any, Dict, Iterable, List


def build_golden_v1_expansion() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    cases.extend(_nlu_cases())
    cases.extend(_dst_cases())
    cases.extend(_tool_cases())
    cases.extend(_rag_cases())
    cases.extend(_e2e_cases())
    if len(cases) != 268:
        raise AssertionError(f"golden expansion must contain 268 cases, got {len(cases)}")
    return cases


def _nlu_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    messages = (
        [f"{prefix}退款 {order_id}" for prefix in (
            "我要", "我想", "申请", "帮我", "请帮我", "需要", "发起", "提交", "办理",
        ) for order_id in ("ORD-1001", "ORD-1002")]
        + [
            "退款申请 ORD-1001", "退款 ORD-1001", "我需要退款 ORD-1001",
            "请办理退款 ORD-1002", "退款订单 ORD-1001",
        ]
    )
    for index, message in enumerate(messages[:18], start=1):
        cases.append(_case(
            "NLU-GX-R", index, "intent", message,
            expected_intent="refund_request",
            expected_slots={"order_id": _order_in(message)},
            risk_level="high",
        ))

    policy_messages = [
        "退款规则是什么", "退款政策是什么", "退款条件是什么", "退款流程是什么",
        "退款期限是多久", "退款时效说明", "退款审核多久", "退款可以申请吗",
    ]
    for index, message in enumerate(policy_messages, start=1):
        cases.append(_case(
            "NLU-GX-P", index, "intent", message,
            expected_intent="refund_policy", risk_level="medium",
        ))

    order_messages = [
        "查询订单 ORD-1001", "查订单 ORD-1002", "帮我查订单 ORD-1001",
        "看看订单 ORD-1002", "订单 ORD-1001 状态", "订单 ORD-1002 详情",
        "查询 ORD-1001 订单", "订单 ORD-1001 怎么样",
    ]
    for index, message in enumerate(order_messages, start=1):
        cases.append(_case(
            "NLU-GX-O", index, "intent", message,
            expected_intent="order_query",
            expected_slots={"order_id": _order_in(message)},
            risk_level="medium",
        ))

    logistics_messages = [
        "查询 ORD-1001 物流", "查物流 ORD-1002", "ORD-1001 快递状态",
        "帮我查 ORD-1002 配送", "订单 ORD-1001 快递到哪了",
        "物流 ORD-1002 什么时候到", "ORD-1001 运单进度",
    ]
    for index, message in enumerate(logistics_messages, start=1):
        cases.append(_case(
            "NLU-GX-L", index, "intent", message,
            expected_intent="logistics_query",
            expected_slots={"order_id": _order_in(message)},
            risk_level="medium",
        ))

    return_messages = [
        "我要退货 ORD-1001", "申请退货 ORD-1002",
        "帮我办理退货 ORD-1001", "需要退货 ORD-1002",
    ]
    for index, message in enumerate(return_messages[:3], start=1):
        cases.append(_case(
            "NLU-GX-RT", index, "intent", message,
            expected_intent="return_request",
            expected_slots={"order_id": _order_in(message)},
            risk_level="high",
        ))

    cancel_messages = [
        "取消订单 ORD-1001", "撤销订单 ORD-1002", "不要这个订单 ORD-1001",
        "订单 ORD-1002 取消",
    ]
    for index, message in enumerate(cancel_messages[:3], start=1):
        cases.append(_case(
            "NLU-GX-C", index, "intent", message,
            expected_intent="cancel_order",
            expected_slots={"order_id": _order_in(message)},
            risk_level="high",
        ))
    assert len(cases) == 47
    return cases


def _dst_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for index, order_id in enumerate(_repeat(("ORD-1001", "ORD-1002"), 12), start=1):
        cases.append(_case(
            "DST-GX-F", index, "slot_fill", turns=["我要退款", f"订单号 {order_id}"],
            expected_intent="refund_request",
            expected_slots={"order_id": order_id}, risk_level="high",
        ))
    for index, (first, corrected) in enumerate(
        _repeat((("ORD-1001", "ORD-1002"), ("ORD-1002", "ORD-1001")), 12),
        start=1,
    ):
        cases.append(_case(
            "DST-GX-C", index, "correction",
            turns=[f"退款 {first}", f"不对，订单号是 {corrected}"],
            expected_intent="refund_request",
            expected_slots={"order_id": corrected}, risk_level="high",
        ))
    for index, order_id in enumerate(_repeat(("ORD-1001", "ORD-1002"), 12), start=1):
        cases.append(_case(
            "DST-GX-S", index, "switch",
            turns=[f"查询订单 {order_id}", "改为申请退款"],
            expected_intent="refund_request",
            expected_slots={"order_id": order_id}, risk_level="high",
        ))
    for index, order_id in enumerate(_repeat(("ORD-1001", "ORD-1002"), 10), start=1):
        cases.append(_case(
            "DST-GX-L", index, "slot_fill", turns=["查询物流", f"订单号 {order_id}"],
            expected_intent="logistics_query",
            expected_slots={"order_id": order_id}, risk_level="medium",
        ))
    for index, order_id in enumerate(_repeat(("ORD-1001", "ORD-1002"), 6), start=1):
        cases.append(_case(
            "DST-GX-RT", index, "slot_fill", turns=["我要退货", f"订单号 {order_id}"],
            expected_intent="return_request",
            expected_slots={"order_id": order_id}, risk_level="high",
        ))
    for index, order_id in enumerate(_repeat(("ORD-1001", "ORD-1002"), 6), start=1):
        cases.append(_case(
            "DST-GX-CO", index, "slot_fill", turns=["取消订单", f"订单号 {order_id}"],
            expected_intent="cancel_order",
            expected_slots={"order_id": order_id}, risk_level="high",
        ))
    assert len(cases) == 58
    return cases


def _tool_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for index, order_id in enumerate(_repeat(("ORD-1001", "ORD-1002", "ORD-9999"), 15), start=1):
        cases.append(_case(
            "TOOL-GX-O", index, "order", f"查询订单 {order_id}",
            expected_tool="query_order", expected_params={"order_id": order_id},
            risk_level="medium",
        ))
    for index, order_id in enumerate(_repeat(("ORD-1001", "ORD-1002"), 15), start=1):
        cases.append(_case(
            "TOOL-GX-L", index, "logistics", f"查询 {order_id} 物流",
            expected_tool="track_package", expected_params={"order_id": order_id},
            expected_tool_trace=["query_order", "track_package"], risk_level="medium",
        ))
    for index in range(1, 11):
        cases.append(_case(
            "TOOL-GX-R", index, "refund", turns=["退款 ORD-1001", "确认"],
            expected_tool="create_refund",
            expected_tool_trace=[
                "query_order", "check_refund_eligibility", "create_refund",
            ],
            expected_postconditions={
                "refund_created": True, "confirmation_observed": True,
            },
            risk_level="critical",
        ))
    for index in range(1, 9):
        cases.append(_case(
            "TOOL-GX-H", index, "handoff",
            turns=["我要投诉，转人工客服", "确认"],
            expected_tool="create_ticket",
            expected_params={"subject": "用户投诉"},
            expected_postconditions={
                "ticket_created": True, "confirmation_observed": True,
            },
            risk_level="high",
        ))
    for index, message in enumerate(_repeat((
        "退款政策是什么", "退款规则是什么", "配送一般几天",
        "金卡有什么优惠", "ERROR-401 怎么处理",
    ), 10), start=1):
        cases.append(_case(
            "TOOL-GX-K", index, "rag", message,
            expected_tool="rag_search", risk_level="medium",
        ))
    assert len(cases) == 58
    return cases


def _rag_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    query_matrix = {
        "refund-window": [
            "退款期限", "购买后几天能退款", "无理由退款时间", "退款条件",
            "退款申请期限", "什么时候不能退款", "退款政策说明", "退款能申请吗",
        ],
        "refund-shipping": [
            "质量问题退货运费", "退货运费谁承担", "质量问题谁付运费",
            "退货邮费说明", "非质量退货运费", "退款运费规则",
            "质量问题退货费用", "商家承担退货运费吗",
        ],
        "logistics-update": [
            "物流更新时间", "物流多久更新", "发货后什么时候有物流",
            "物流信息没更新", "配送状态刷新时间", "订单物流更新",
            "物流何时显示", "快递信息更新",
        ],
        "error-401": [
            "ERROR-401", "401 登录失败", "认证失败怎么处理", "登录报401",
            "ERROR-401 是什么", "401 错误", "登录认证失败", "401 解决办法",
        ],
        "error-500": [
            "ERROR-500", "500 服务错误", "服务器500怎么办", "HTTP 500",
            "500错误要记录什么", "服务端错误", "500 一直出现", "500 排查",
        ],
        "points-expiry": [
            "积分有效期", "积分什么时候过期", "积分清零", "积分能用多久",
            "积分过期怎么办", "会员积分有效期", "积分规则", "积分到期",
        ],
        "membership-gold": [
            "金卡折扣", "金卡打几折", "金卡会员优惠", "金卡权益",
            "金卡价格", "会员折扣", "金卡能优惠吗", "金卡特价商品",
        ],
        "delivery-express": [
            "加急配送费用", "加急快递多少钱", "次日达收费", "配送加急价格",
            "标准配送免费吗", "加急配送收费", "加急运费", "配送费用说明",
        ],
    }
    for chunk_id, queries in query_matrix.items():
        for index, query in enumerate(queries, start=1):
            cases.append(_case(
                f"RAG-GX-{chunk_id.upper()}", index, "faq", query=query,
                relevant_ids=[chunk_id], required_evidence_ids=[chunk_id],
                must_abstain=False, risk_level="medium",
            ))
    for index, query in enumerate((
        "火星配送政策", "月球配送费用", "保证今天退款到账吗", "给我其他用户订单",
    ), start=1):
        cases.append(_case(
            "RAG-GX-NA", index, "no_answer", query=query,
            relevant_ids=[], must_abstain=True, risk_level="high",
        ))
    assert len(cases) == 68
    return cases


def _e2e_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for index in range(1, 11):
        cases.append(_case(
            "E2E-GX-R", index, "refund_success",
            turns=["退款 ORD-1001", "确认"], expected_status="completed",
            expected_tool_trace=[
                "query_order", "check_refund_eligibility", "create_refund",
            ],
            expected_postconditions={
                "refund_created": True, "confirmation_observed": True,
            },
            risk_level="critical",
        ))
    for index in range(1, 9):
        cases.append(_case(
            "E2E-GX-G", index, "confirmation_gate",
            turns=["退款 ORD-1001"], expected_status="awaiting_user",
            forbidden_tools=["create_refund"],
            expected_postconditions={"refund_created": False},
            risk_level="critical",
        ))
    for index, rejection in enumerate(
        ("取消", "不办了", "不要了", "算了", "先别提交", "不要退款了"),
        start=1,
    ):
        cases.append(_case(
            "E2E-GX-X", index, "rejection_safety",
            turns=["退款 ORD-1001", rejection], expected_status="completed",
            forbidden_tools=["create_refund"],
            expected_postconditions={"refund_created": False},
            risk_level="critical",
        ))
    for index, order_id in enumerate(
        ("ORD-EXPIRED", "ORD-9999", "ORD-EXPIRED", "ORD-9999"), start=1,
    ):
        cases.append(_case(
            "E2E-GX-B", index, "refund_boundary",
            turns=[f"退款 {order_id}"], expected_status="completed",
            forbidden_tools=["create_refund"],
            expected_postconditions={"refund_created": False},
            risk_level="high",
        ))
    for index, turns in enumerate((
        ["查询 ORD-1001"], ["查询 ORD-1002 物流"], ["查 ORD-1002 订单和物流"],
        ["退款政策是什么"], ["查询 ORD-9999"],
    ), start=1):
        cases.append(_case(
            "E2E-GX-Q", index, "read_workflow", turns=turns,
            expected_status="completed", risk_level="medium",
        ))
    for index, (turns, failed_tool) in enumerate((
        (["退款 ORD-1001"], "query_order"),
        (["查询 ORD-1002 物流"], "track_package"),
        (["我要投诉，转人工客服", "确认"], "create_ticket"),
        (["查询 ORD-1001"], "query_order"),
    ), start=1):
        cases.append(_case(
            "E2E-GX-F", index, "tool_failure", turns=turns,
            fail_tools=[failed_tool], expected_status="failed",
            forbidden_tools=["create_refund"] if failed_tool == "query_order" else [],
            risk_level="high",
        ))
    assert len(cases) == 37
    return cases


def _case(
    prefix: str,
    index: int,
    category: str,
    message: str | None = None,
    **values: Any,
) -> Dict[str, Any]:
    case: Dict[str, Any] = {
        "id": f"{prefix}-{index:03d}",
        "category": category,
        "source": "golden_v1_scenario_matrix",
        "review_status": "generated_pending_human_calibration",
    }
    if message is not None:
        case["message"] = message
    case.update(values)
    return case


def _repeat(values: Iterable[Any], count: int) -> List[Any]:
    iterator = cycle(values)
    return [next(iterator) for _ in range(count)]


def _order_in(message: str) -> str:
    return "ORD-1002" if "ORD-1002" in message else "ORD-1001"
