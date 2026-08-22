"""Explicit ToolRegistry adapters for internal customer-service RPC clients."""
from __future__ import annotations

from typing import Any, Dict, Iterable

from core.internal_rpc import InternalRPCClients
from core.tool_names import canonical_tool_name
from core.tool_registry import LocalToolAdapter, ToolRegistry, ToolSpec, ToolType


def register_internal_rpc_tools(
    registry: ToolRegistry,
    clients: InternalRPCClients,
    action_repository: Any,
) -> None:
    """Register typed internal RPC operations without MCP discovery."""
    definitions = [
        ("query_order", clients.commerce, ("order_id",), ToolType.READ),
        ("query_order_items", clients.commerce, ("order_id",), ToolType.READ),
        ("query_payment_detail", clients.commerce, ("order_id",), ToolType.READ),
        ("query_invoice", clients.commerce, ("order_id",), ToolType.READ),
        (
            "track_package",
            clients.fulfillment,
            (),
            ToolType.READ,
            ("order_id", "tracking_no"),
        ),
        (
            "estimate_delivery",
            clients.fulfillment,
            (),
            ToolType.READ,
            ("order_id", "tracking_no"),
        ),
        (
            "diagnose_delivery_exception",
            clients.fulfillment,
            (),
            ToolType.READ,
            ("order_id", "tracking_no"),
        ),
        (
            "check_refund_eligibility",
            clients.after_sales,
            ("order_id",),
            ToolType.READ,
            ("order_id", "reason"),
        ),
        (
            "evaluate_after_sales_options",
            clients.after_sales,
            ("order_id",),
            ToolType.READ,
        ),
        (
            "calculate_refund_quote",
            clients.after_sales,
            ("order_id",),
            ToolType.READ,
        ),
        (
            "create_refund",
            clients.after_sales,
            ("order_id", "action_id"),
            ToolType.WRITE,
            ("order_id", "action_id", "reason"),
        ),
        (
            "create_return",
            clients.after_sales,
            ("order_id", "action_id"),
            ToolType.WRITE,
            ("order_id", "action_id", "reason"),
        ),
        (
            "cancel_order",
            clients.after_sales,
            ("order_id", "action_id"),
            ToolType.WRITE,
            ("order_id", "action_id", "reason"),
        ),
        (
            "create_ticket",
            clients.after_sales,
            ("subject", "description", "action_id"),
            ToolType.WRITE,
            ("subject", "description", "action_id", "order_id"),
        ),
    ]
    for definition in definitions:
        logical_name, client, required, tool_type, *optional_fields = definition
        fields = optional_fields[0] if optional_fields else required
        _register_rpc_tool(
            registry,
            client,
            logical_name,
            required=required,
            fields=fields,
            tool_type=tool_type,
            action_repository=action_repository,
        )


def _register_rpc_tool(
    registry: ToolRegistry,
    client: Any,
    logical_name: str,
    *,
    required: Iterable[str],
    fields: Iterable[str],
    tool_type: ToolType,
    action_repository: Any,
) -> None:
    name = canonical_tool_name(logical_name)
    required_fields = list(required)
    property_names = list(dict.fromkeys([*fields, *required_fields]))
    spec = ToolSpec(
        name=name,
        description=f"Internal RPC operation: {logical_name}",
        input_schema={
            "type": "object",
            "properties": {
                field: {"type": "string"}
                for field in property_names
            },
            "required": required_fields,
            "additionalProperties": False,
        },
        output_schema={"type": "object"},
        tool_type=tool_type,
        timeout_s=30,
    )
    method = getattr(client, logical_name)

    async def handler(
        params: Dict[str, Any],
        context: Any,
        *,
        _method=method,
        _spec=spec,
    ) -> Any:
        arguments = dict(params)
        if _spec.tool_type == ToolType.READ:
            return await _method(**arguments)
        action_id = str(arguments["action_id"])
        return await action_repository.execute(
            _spec.name,
            action_id,
            arguments,
            lambda: _method(**arguments),
            context=context if isinstance(context, dict) else None,
        )

    registry.register(LocalToolAdapter(spec, handler))
