"""Canonical names for tools exposed through domain MCP servers."""

from typing import Dict


TOOL_NAMES: Dict[str, str] = {
    "query_order": "commerce.query_order",
    "query_order_items": "commerce.query_order_items",
    "query_payment_detail": "commerce.query_payment_detail",
    "query_invoice": "commerce.query_invoice",
    "track_package": "fulfillment.track_package",
    "estimate_delivery": "fulfillment.estimate_delivery",
    "diagnose_delivery_exception": "fulfillment.diagnose_delivery_exception",
    "check_refund_eligibility": "after_sales.check_refund_eligibility",
    "evaluate_after_sales_options": "after_sales.evaluate_after_sales_options",
    "calculate_refund_quote": "after_sales.calculate_refund_quote",
    "create_refund": "after_sales.create_refund",
    "create_return": "after_sales.create_return",
    "cancel_order": "after_sales.cancel_order",
    "create_ticket": "after_sales.create_ticket",
    "rag_search": "knowledge.rag_search",
}


def canonical_tool_name(name: str) -> str:
    """Return the domain-qualified name for a logical tool name."""
    return TOOL_NAMES.get(name, name)


def logical_tool_name(name: str) -> str:
    """Return the unqualified name used by business completion rules."""
    return name.rsplit(".", 1)[-1]
