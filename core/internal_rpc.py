"""Internal customer-service RPC contracts and deterministic mock clients."""
from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Optional, Protocol


class CommerceRPC(Protocol):
    async def query_order(self, order_id: str) -> Dict[str, Any]: ...
    async def query_order_items(self, order_id: str) -> Dict[str, Any]: ...
    async def query_payment_detail(self, order_id: str) -> Dict[str, Any]: ...
    async def query_invoice(self, order_id: str) -> Dict[str, Any]: ...


class FulfillmentRPC(Protocol):
    async def track_package(
        self,
        order_id: Optional[str] = None,
        tracking_no: Optional[str] = None,
    ) -> Dict[str, Any]: ...
    async def estimate_delivery(
        self,
        order_id: Optional[str] = None,
        tracking_no: Optional[str] = None,
    ) -> Dict[str, Any]: ...
    async def diagnose_delivery_exception(
        self,
        order_id: Optional[str] = None,
        tracking_no: Optional[str] = None,
    ) -> Dict[str, Any]: ...


class AfterSalesRPC(Protocol):
    async def check_refund_eligibility(
        self,
        order_id: str,
        reason: str = "user_requested",
    ) -> Dict[str, Any]: ...
    async def evaluate_after_sales_options(
        self,
        order_id: str,
    ) -> Dict[str, Any]: ...
    async def calculate_refund_quote(self, order_id: str) -> Dict[str, Any]: ...
    async def create_refund(
        self,
        order_id: str,
        action_id: str,
        reason: str = "user_requested",
    ) -> Dict[str, Any]: ...
    async def create_return(
        self,
        order_id: str,
        action_id: str,
        reason: str = "user_requested",
    ) -> Dict[str, Any]: ...
    async def cancel_order(
        self,
        order_id: str,
        action_id: str,
        reason: str = "user_requested",
    ) -> Dict[str, Any]: ...
    async def create_ticket(
        self,
        subject: str,
        description: str,
        action_id: str,
        order_id: Optional[str] = None,
    ) -> Dict[str, Any]: ...


@dataclass(frozen=True)
class InternalRPCClients:
    commerce: CommerceRPC
    fulfillment: FulfillmentRPC
    after_sales: AfterSalesRPC


_ORDERS: Dict[str, Dict[str, Any]] = {
    "ORD-1001": {
        "order_id": "ORD-1001",
        "status": "delivered",
        "amount": 299.0,
        "currency": "CNY",
        "paid_at": "2026-07-30",
        "refundable_until": "2099-08-08",
        "items": [{"sku_id": "SKU-1001", "name": "无线耳机", "quantity": 1}],
        "payment_method": "wechat_pay",
        "payment_status": "paid",
        "invoice_status": "issued",
    },
    "ORD-1002": {
        "order_id": "ORD-1002",
        "status": "in_transit",
        "amount": 89.0,
        "currency": "CNY",
        "paid_at": "2026-07-28",
        "refundable_until": None,
        "items": [{"sku_id": "SKU-1002", "name": "手机壳", "quantity": 1}],
        "payment_method": "alipay",
        "payment_status": "paid",
        "invoice_status": "not_requested",
    },
    "ORD-EXPIRED": {
        "order_id": "ORD-EXPIRED",
        "status": "delivered",
        "amount": 49.0,
        "currency": "CNY",
        "paid_at": "2026-06-01",
        "refundable_until": "2026-06-10",
        "items": [{"sku_id": "SKU-1003", "name": "数据线", "quantity": 1}],
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


class MockCustomerServiceRPC(CommerceRPC, FulfillmentRPC, AfterSalesRPC):
    """In-process RPC mock with no mutable idempotency state."""

    async def query_order(self, order_id: str) -> Dict[str, Any]:
        order = _ORDERS.get(order_id)
        if order is None:
            return {"found": False, "order_id": order_id, "error": "order_not_found"}
        return {"found": True, **deepcopy(order)}

    async def query_order_items(self, order_id: str) -> Dict[str, Any]:
        order = await self.query_order(order_id)
        if not order["found"]:
            return order
        return {"found": True, "order_id": order_id, "items": order["items"]}

    async def query_payment_detail(self, order_id: str) -> Dict[str, Any]:
        order = await self.query_order(order_id)
        if not order["found"]:
            return order
        return {
            "found": True,
            "order_id": order_id,
            "amount": order["amount"],
            "currency": order["currency"],
            "paid_at": order["paid_at"],
            "payment_method": order["payment_method"],
            "payment_status": order["payment_status"],
        }

    async def query_invoice(self, order_id: str) -> Dict[str, Any]:
        order = await self.query_order(order_id)
        if not order["found"]:
            return order
        return {
            "found": True,
            "order_id": order_id,
            "invoice_status": order["invoice_status"],
        }

    async def track_package(
        self,
        order_id: Optional[str] = None,
        tracking_no: Optional[str] = None,
    ) -> Dict[str, Any]:
        tracking = _LOGISTICS.get(order_id or "")
        if tracking is None and tracking_no:
            tracking = next(
                (
                    value for value in _LOGISTICS.values()
                    if value["tracking_no"] == tracking_no
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

    async def estimate_delivery(
        self,
        order_id: Optional[str] = None,
        tracking_no: Optional[str] = None,
    ) -> Dict[str, Any]:
        tracking = await self.track_package(order_id, tracking_no)
        if not tracking["found"]:
            return tracking
        return {
            "found": True,
            "order_id": tracking["order_id"],
            "tracking_no": tracking["tracking_no"],
            "status": tracking["status"],
            "estimated_delivery": tracking["estimated_delivery"],
        }

    async def diagnose_delivery_exception(
        self,
        order_id: Optional[str] = None,
        tracking_no: Optional[str] = None,
    ) -> Dict[str, Any]:
        tracking = await self.track_package(order_id, tracking_no)
        if not tracking["found"]:
            return {
                **tracking,
                "exception_type": "tracking_not_available",
                "recommended_action": "create_delivery_ticket",
            }
        delivered = tracking["status"] == "delivered"
        return {
            "found": True,
            "order_id": tracking["order_id"],
            "tracking_no": tracking["tracking_no"],
            "exception_type": None if delivered else "in_transit",
            "recommended_action": (
                "confirm_receipt" if delivered else "wait_for_next_scan"
            ),
        }

    async def check_refund_eligibility(
        self,
        order_id: str,
        reason: str = "user_requested",
    ) -> Dict[str, Any]:
        order = await self.query_order(order_id)
        if not order["found"]:
            return {"eligible": False, "order_id": order_id, "reason": "order_not_found"}
        refundable_until = order.get("refundable_until")
        eligible = (
            order["status"] == "delivered"
            and refundable_until is not None
            and date.today() <= date.fromisoformat(refundable_until)
        )
        return {
            "eligible": eligible,
            "order_id": order_id,
            "refund_reason": reason,
            "reason": "within_refund_window" if eligible else "outside_refund_window",
            "refundable_until": refundable_until,
        }

    async def evaluate_after_sales_options(
        self,
        order_id: str,
    ) -> Dict[str, Any]:
        order = await self.query_order(order_id)
        if not order["found"]:
            return {
                "found": False,
                "order_id": order_id,
                "available_actions": [],
                "reason": "order_not_found",
            }
        refund = await self.check_refund_eligibility(order_id)
        actions = ["refund", "return"] if refund["eligible"] else []
        if order["status"] in {"paid", "processing"}:
            actions.append("cancel")
        return {
            "found": True,
            "order_id": order_id,
            "available_actions": actions,
            "refund_eligible": refund["eligible"],
            "return_eligible": refund["eligible"],
            "cancel_eligible": "cancel" in actions,
            "reason": refund["reason"],
        }

    async def calculate_refund_quote(self, order_id: str) -> Dict[str, Any]:
        order = await self.query_order(order_id)
        if not order["found"]:
            return order
        eligibility = await self.check_refund_eligibility(order_id)
        return {
            "found": True,
            "eligible": eligibility["eligible"],
            "order_id": order_id,
            "refund_amount": order["amount"] if eligibility["eligible"] else 0.0,
            "currency": order["currency"],
            "reason": eligibility["reason"],
        }

    async def create_refund(
        self,
        order_id: str,
        action_id: str,
        reason: str = "user_requested",
    ) -> Dict[str, Any]:
        eligibility = await self.check_refund_eligibility(order_id, reason)
        if not eligibility["eligible"]:
            return {
                "created": False,
                "order_id": order_id,
                "action_id": action_id,
                "error": eligibility["reason"],
            }
        return {
            "created": True,
            "refund_id": _stable_id("REF", action_id),
            "order_id": order_id,
            "action_id": action_id,
            "status": "submitted",
            "idempotent_replay": False,
        }

    async def create_return(
        self,
        order_id: str,
        action_id: str,
        reason: str = "user_requested",
    ) -> Dict[str, Any]:
        eligibility = await self.check_refund_eligibility(order_id, reason)
        if not eligibility["eligible"]:
            return {
                "created": False,
                "order_id": order_id,
                "action_id": action_id,
                "error": eligibility["reason"],
            }
        return {
            "created": True,
            "return_id": _stable_id("RET", action_id),
            "order_id": order_id,
            "action_id": action_id,
            "status": "submitted",
        }

    async def cancel_order(
        self,
        order_id: str,
        action_id: str,
        reason: str = "user_requested",
    ) -> Dict[str, Any]:
        del reason
        order = await self.query_order(order_id)
        cancellable = order.get("status") in {"paid", "processing"}
        return {
            "created": cancellable,
            "cancellation_id": _stable_id("CAN", action_id) if cancellable else None,
            "order_id": order_id,
            "action_id": action_id,
            "error": None if cancellable else "order_not_cancellable",
        }

    async def create_ticket(
        self,
        subject: str,
        description: str,
        action_id: str,
        order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "created": True,
            "ticket_id": _stable_id("TKT", action_id),
            "subject": subject,
            "description": description,
            "order_id": order_id,
            "action_id": action_id,
            "status": "open",
        }


def build_mock_rpc_clients() -> InternalRPCClients:
    client = MockCustomerServiceRPC()
    return InternalRPCClients(
        commerce=client,
        fulfillment=client,
        after_sales=client,
    )


def _stable_id(prefix: str, action_id: str) -> str:
    digest = hashlib.sha256(action_id.encode("utf-8")).hexdigest()[:10].upper()
    return f"{prefix}-{digest}"
