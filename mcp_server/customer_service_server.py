"""Standard MCP server exposing deterministic customer-service tools."""

from copy import deepcopy
from datetime import date
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
        "refundable_until": "2026-08-08",
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
_TICKETS_BY_ACTION: Dict[str, Dict[str, Any]] = {}


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
def track_package(order_id: str) -> Dict[str, Any]:
    """Query the deterministic package tracking timeline for an order."""
    order = _order_or_error(order_id)
    if not order["found"]:
        return order
    tracking = _LOGISTICS.get(order_id)
    if tracking is None:
        return {
            "found": False,
            "order_id": order_id,
            "error": "tracking_not_available",
        }
    return {"found": True, **deepcopy(tracking)}


@server.tool()
def check_refund_eligibility(
    order_id: str,
    reason: str = "user_requested",
    as_of: str = "2026-08-02",
) -> Dict[str, Any]:
    """Check whether the mock order can be refunded as of an ISO date."""
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
        and date.fromisoformat(as_of) <= date.fromisoformat(refundable_until)
    )
    return {
        "eligible": eligible,
        "order_id": order_id,
        "refund_reason": reason,
        "reason": "within_refund_window" if eligible else "outside_refund_window",
        "refundable_until": refundable_until,
    }


@server.tool()
def create_refund(
    order_id: str,
    action_id: str,
    reason: str = "user_requested",
) -> Dict[str, Any]:
    """Create an idempotent mock refund keyed by action_id."""
    existing = _REFUNDS_BY_ACTION.get(action_id)
    if existing is not None:
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
    return deepcopy(refund)


@server.tool()
def create_ticket(
    subject: str,
    description: str,
    action_id: str,
    order_id: str | None = None,
) -> Dict[str, Any]:
    """Create an idempotent mock handoff ticket keyed by action_id."""
    existing = _TICKETS_BY_ACTION.get(action_id)
    if existing is not None:
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
    return deepcopy(ticket)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
