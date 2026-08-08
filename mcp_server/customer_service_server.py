"""Standard MCP server exposing deterministic customer-service tools."""

from copy import deepcopy
from datetime import date
import hashlib
import json
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP


server = FastMCP(
    "ggbot-customer-service",
    instructions="Deterministic mock order, logistics, refund, and ticket tools.",
)

_ORDERS: Dict[str, Dict[str, Any]] = {
    "ORD-1001": {
        "order_id": "ORD-1001",
        "user_id": "user-001",
        "status": "delivered",
        "amount": 299.0,
        "currency": "CNY",
        "paid_at": "2026-07-30",
        "delivered_at": "2026-08-01",
        "refundable_until": "2099-08-08",
        "items": [
            {"sku_id": "SKU-1001", "name": "无线耳机", "quantity": 1},
        ],
        "payment_method": "wechat_pay",
        "payment_status": "paid",
        "invoice_status": "issued",
    },
    "ORD-1002": {
        "order_id": "ORD-1002",
        "user_id": "user-002",
        "status": "in_transit",
        "amount": 89.0,
        "currency": "CNY",
        "paid_at": "2026-07-28",
        "delivered_at": None,
        "refundable_until": None,
        "items": [
            {"sku_id": "SKU-1002", "name": "手机壳", "quantity": 1},
        ],
        "payment_method": "alipay",
        "payment_status": "paid",
        "invoice_status": "not_requested",
    },
    "ORD-EXPIRED": {
        "order_id": "ORD-EXPIRED",
        "user_id": "user-003",
        "status": "delivered",
        "amount": 49.0,
        "currency": "CNY",
        "paid_at": "2026-06-01",
        "delivered_at": "2026-06-03",
        "refundable_until": "2026-06-10",
        "items": [
            {"sku_id": "SKU-1003", "name": "数据线", "quantity": 1},
        ],
        "payment_method": "bank_card",
        "payment_status": "paid",
        "invoice_status": "not_requested",
    },
}

_LOGISTICS: Dict[str, Dict[str, Any]] = {
    "ORD-1001": {
        "order_id": "ORD-1001",
        "tracking_no": "SF1001001",
        "status": "delivered",
        "estimated_delivery": "2026-08-01",
        "events": [
            {"time": "2026-07-31T09:00:00", "description": "Package dispatched"},
            {"time": "2026-08-01T15:30:00", "description": "Package delivered"},
        ],
    },
    "ORD-1002": {
        "order_id": "ORD-1002",
        "tracking_no": "SF1001002",
        "status": "in_transit",
        "estimated_delivery": "2026-08-04",
        "events": [
            {"time": "2026-08-01T12:00:00", "description": "Package dispatched"},
            {"time": "2026-08-02T08:20:00", "description": "Arrived at sorting center"},
        ],
    },
}

_REFUNDS_BY_ACTION: Dict[str, Dict[str, Any]] = {}
_RETURNS_BY_ACTION: Dict[str, Dict[str, Any]] = {}
_CANCELLATIONS_BY_ACTION: Dict[str, Dict[str, Any]] = {}
_TICKETS_BY_ACTION: Dict[str, Dict[str, Any]] = {}
_REFUND_FINGERPRINTS: Dict[str, str] = {}
_RETURN_FINGERPRINTS: Dict[str, str] = {}
_CANCELLATION_FINGERPRINTS: Dict[str, str] = {}
_TICKET_FINGERPRINTS: Dict[str, str] = {}


def _current_date() -> date:
    """Return the business date; tests replace this clock deterministically."""
    return date.today()


def _payload_fingerprint(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _order_or_error(order_id: str) -> Dict[str, Any]:
    order = _ORDERS.get(order_id)
    if order is None:
        return {"found": False, "order_id": order_id, "error": "order_not_found"}
    return {"found": True, **deepcopy(order)}


@server.tool()
def query_order(order_id: str) -> Dict[str, Any]:
    """Query order status, amount, payment, delivery, and refund dates."""
    return _order_or_error(order_id)


@server.tool()
def query_order_items(order_id: str) -> Dict[str, Any]:
    """Query item lines for an order."""
    order = _order_or_error(order_id)
    if not order["found"]:
        return order
    return {
        "found": True,
        "order_id": order_id,
        "items": deepcopy(order.get("items", [])),
    }


@server.tool()
def query_payment_detail(order_id: str) -> Dict[str, Any]:
    """Query payment method and payment status for an order."""
    order = _order_or_error(order_id)
    if not order["found"]:
        return order
    return {
        "found": True,
        "order_id": order_id,
        "amount": order["amount"],
        "currency": order["currency"],
        "paid_at": order.get("paid_at"),
        "payment_method": order.get("payment_method"),
        "payment_status": order.get("payment_status"),
    }


@server.tool()
def query_invoice(order_id: str) -> Dict[str, Any]:
    """Query invoice status for an order."""
    order = _order_or_error(order_id)
    if not order["found"]:
        return order
    return {
        "found": True,
        "order_id": order_id,
        "invoice_status": order.get("invoice_status"),
    }


@server.tool()
def track_package(
    order_id: str | None = None,
    tracking_no: str | None = None,
) -> Dict[str, Any]:
    """Query a package timeline by order ID or tracking number."""
    if order_id is None and tracking_no is None:
        return {
            "found": False,
            "error": "order_id_or_tracking_no_required",
        }
    if order_id is not None:
        order = _order_or_error(order_id)
        if not order["found"]:
            return order
        tracking = _LOGISTICS.get(order_id)
    else:
        tracking = next(
            (
                item for item in _LOGISTICS.values()
                if item.get("tracking_no") == tracking_no
            ),
            None,
        )
    if tracking is None:
        return {
            "found": False,
            "order_id": order_id,
            "tracking_no": tracking_no,
            "error": "tracking_not_available",
        }
    return {"found": True, **deepcopy(tracking)}


@server.tool()
def estimate_delivery(
    order_id: str | None = None,
    tracking_no: str | None = None,
) -> Dict[str, Any]:
    """Return the current delivery estimate for an order or tracking number."""
    tracking = track_package(order_id=order_id, tracking_no=tracking_no)
    if not tracking["found"]:
        return tracking
    return {
        "found": True,
        "order_id": tracking.get("order_id"),
        "tracking_no": tracking.get("tracking_no"),
        "status": tracking.get("status"),
        "estimated_delivery": tracking.get("estimated_delivery"),
    }


@server.tool()
def diagnose_delivery_exception(
    order_id: str | None = None,
    tracking_no: str | None = None,
) -> Dict[str, Any]:
    """Diagnose a stalled, missing, or completed package timeline."""
    tracking = track_package(order_id=order_id, tracking_no=tracking_no)
    if not tracking["found"]:
        return {
            **tracking,
            "exception_type": "tracking_not_available",
            "recommended_action": "create_delivery_ticket",
        }
    status = tracking.get("status")
    if status == "delivered":
        exception_type = None
        recommended_action = "confirm_receipt"
    elif len(tracking.get("events", [])) < 2:
        exception_type = "insufficient_tracking_events"
        recommended_action = "create_delivery_ticket"
    else:
        exception_type = "in_transit"
        recommended_action = "wait_for_next_scan"
    return {
        "found": True,
        "order_id": tracking.get("order_id"),
        "tracking_no": tracking.get("tracking_no"),
        "exception_type": exception_type,
        "recommended_action": recommended_action,
    }


@server.tool()
def check_refund_eligibility(
    order_id: str,
    reason: str = "user_requested",
) -> Dict[str, Any]:
    """Check whether the mock order can be refunded on the business date."""
    order = _order_or_error(order_id)
    if not order["found"]:
        return {
            "eligible": False,
            "order_id": order_id,
            "reason": "order_not_found",
        }

    refundable_until = order.get("refundable_until")
    eligible = (
        order["status"] == "delivered"
        and refundable_until is not None
        and _current_date() <= date.fromisoformat(refundable_until)
    )
    return {
        "eligible": eligible,
        "order_id": order_id,
        "refund_reason": reason,
        "reason": "within_refund_window" if eligible else "outside_refund_window",
        "refundable_until": refundable_until,
    }


@server.tool()
def evaluate_after_sales_options(order_id: str) -> Dict[str, Any]:
    """Return the currently available refund, return, and cancellation options."""
    order = _order_or_error(order_id)
    if not order["found"]:
        return {
            "found": False,
            "order_id": order_id,
            "available_actions": [],
            "reason": "order_not_found",
        }
    refund = check_refund_eligibility(order_id)
    available_actions = []
    if refund["eligible"]:
        available_actions.extend(["refund", "return"])
    if order["status"] in {"paid", "processing"}:
        available_actions.append("cancel")
    return {
        "found": True,
        "order_id": order_id,
        "available_actions": available_actions,
        "refund_eligible": refund["eligible"],
        "return_eligible": refund["eligible"],
        "cancel_eligible": "cancel" in available_actions,
        "reason": refund["reason"],
    }


@server.tool()
def calculate_refund_quote(order_id: str) -> Dict[str, Any]:
    """Calculate the currently refundable amount without creating a refund."""
    order = _order_or_error(order_id)
    if not order["found"]:
        return order
    eligibility = check_refund_eligibility(order_id)
    return {
        "found": True,
        "eligible": eligibility["eligible"],
        "order_id": order_id,
        "refund_amount": order["amount"] if eligibility["eligible"] else 0.0,
        "currency": order["currency"],
        "reason": eligibility["reason"],
    }


@server.tool()
def create_refund(
    order_id: str,
    action_id: str,
    reason: str = "user_requested",
) -> Dict[str, Any]:
    """Create an idempotent mock refund keyed by action_id."""
    fingerprint = _payload_fingerprint({
        "order_id": order_id,
        "reason": reason,
    })
    existing = _REFUNDS_BY_ACTION.get(action_id)
    if existing is not None:
        if _REFUND_FINGERPRINTS.get(action_id) != fingerprint:
            return {
                "created": False,
                "action_id": action_id,
                "error": "idempotency_conflict",
            }
        return {**deepcopy(existing), "idempotent_replay": True}

    eligibility = check_refund_eligibility(order_id=order_id, reason=reason)
    if not eligibility["eligible"]:
        return {
            "created": False,
            "order_id": order_id,
            "action_id": action_id,
            "error": eligibility["reason"],
        }

    refund = {
        "created": True,
        "refund_id": f"REF-{len(_REFUNDS_BY_ACTION) + 1:04d}",
        "order_id": order_id,
        "action_id": action_id,
        "reason": reason,
        "status": "submitted",
        "idempotent_replay": False,
    }
    _REFUNDS_BY_ACTION[action_id] = refund
    _REFUND_FINGERPRINTS[action_id] = fingerprint
    return deepcopy(refund)


@server.tool()
def create_return(
    order_id: str,
    action_id: str,
    reason: str = "user_requested",
) -> Dict[str, Any]:
    """Create an idempotent return request keyed by action_id."""
    fingerprint = _payload_fingerprint({
        "order_id": order_id,
        "reason": reason,
    })
    existing = _RETURNS_BY_ACTION.get(action_id)
    if existing is not None:
        if _RETURN_FINGERPRINTS.get(action_id) != fingerprint:
            return {
                "created": False,
                "action_id": action_id,
                "error": "idempotency_conflict",
            }
        return {**deepcopy(existing), "idempotent_replay": True}
    eligibility = check_refund_eligibility(order_id=order_id, reason=reason)
    if not eligibility["eligible"]:
        return {
            "created": False,
            "order_id": order_id,
            "action_id": action_id,
            "error": eligibility["reason"],
        }
    return_request = {
        "created": True,
        "return_id": f"RET-{len(_RETURNS_BY_ACTION) + 1:04d}",
        "order_id": order_id,
        "action_id": action_id,
        "reason": reason,
        "status": "submitted",
        "idempotent_replay": False,
    }
    _RETURNS_BY_ACTION[action_id] = return_request
    _RETURN_FINGERPRINTS[action_id] = fingerprint
    return deepcopy(return_request)


@server.tool()
def cancel_order(
    order_id: str,
    action_id: str,
    reason: str = "user_requested",
) -> Dict[str, Any]:
    """Cancel an eligible order idempotently."""
    fingerprint = _payload_fingerprint({
        "order_id": order_id,
        "reason": reason,
    })
    existing = _CANCELLATIONS_BY_ACTION.get(action_id)
    if existing is not None:
        if _CANCELLATION_FINGERPRINTS.get(action_id) != fingerprint:
            return {
                "cancelled": False,
                "action_id": action_id,
                "error": "idempotency_conflict",
            }
        return {**deepcopy(existing), "idempotent_replay": True}
    order = _order_or_error(order_id)
    if not order["found"] or order.get("status") not in {"paid", "processing"}:
        return {
            "cancelled": False,
            "order_id": order_id,
            "action_id": action_id,
            "error": "order_not_cancellable",
        }
    cancellation = {
        "cancelled": True,
        "order_id": order_id,
        "action_id": action_id,
        "reason": reason,
        "status": "cancelled",
        "idempotent_replay": False,
    }
    _CANCELLATIONS_BY_ACTION[action_id] = cancellation
    _CANCELLATION_FINGERPRINTS[action_id] = fingerprint
    return deepcopy(cancellation)


@server.tool()
def create_ticket(
    subject: str,
    description: str,
    action_id: str,
    order_id: str | None = None,
) -> Dict[str, Any]:
    """Create an idempotent mock handoff ticket keyed by action_id."""
    fingerprint = _payload_fingerprint({
        "subject": subject,
        "description": description,
        "order_id": order_id,
    })
    existing = _TICKETS_BY_ACTION.get(action_id)
    if existing is not None:
        if _TICKET_FINGERPRINTS.get(action_id) != fingerprint:
            return {
                "created": False,
                "action_id": action_id,
                "error": "idempotency_conflict",
            }
        return {**deepcopy(existing), "idempotent_replay": True}

    ticket = {
        "created": True,
        "ticket_id": f"TKT-{len(_TICKETS_BY_ACTION) + 1:04d}",
        "action_id": action_id,
        "order_id": order_id,
        "subject": subject,
        "description": description,
        "status": "open",
        "idempotent_replay": False,
    }
    _TICKETS_BY_ACTION[action_id] = ticket
    _TICKET_FINGERPRINTS[action_id] = fingerprint
    return deepcopy(ticket)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
