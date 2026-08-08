import asyncio
import os
import sys
from datetime import date
from pathlib import Path

from mcp import ClientSession, StdioServerParameters

from core.mcp_adapter import MCPClient, MCPToolAdapter
from core.tool_registry import ToolRegistry, ToolType
from mcp_server import customer_service_server


ROOT = Path(__file__).parent.parent.resolve()


def test_official_mcp_top_level_imports_remain_available():
    assert ClientSession is not None
    assert StdioServerParameters is not None


def run(coro):
    return asyncio.run(coro)


def server_client() -> MCPClient:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    return MCPClient(
        command=sys.executable,
        args=["-m", "mcp_server.customer_service_server"],
        env=env,
    )


async def protocol_scenario():
    async with server_client() as client:
        tools = await client.list_tools()
        names = {tool.name for tool in tools}
        assert names == {
            "query_order",
            "query_order_items",
            "query_payment_detail",
            "query_invoice",
            "track_package",
            "estimate_delivery",
            "diagnose_delivery_exception",
            "check_refund_eligibility",
            "evaluate_after_sales_options",
            "calculate_refund_quote",
            "create_refund",
            "create_return",
            "cancel_order",
            "create_ticket",
        }

        order = await client.call_tool("query_order", {"order_id": "ORD-1001"})
        assert order["found"] is True
        assert order["status"] == "delivered"

        tracking = await client.call_tool(
            "track_package", {"order_id": "ORD-1002"}
        )
        assert tracking["found"] is True
        assert tracking["tracking_no"] == "SF1001002"

        tracking_by_number = await client.call_tool(
            "track_package",
            {"tracking_no": "SF1001002"},
        )
        assert tracking_by_number["found"] is True
        assert tracking_by_number["order_id"] == "ORD-1002"

        eligibility = await client.call_tool(
            "check_refund_eligibility",
            {"order_id": "ORD-1001", "reason": "quality_issue"},
        )
        assert eligibility["eligible"] is True

        items = await client.call_tool(
            "query_order_items",
            {"order_id": "ORD-1001"},
        )
        assert items["items"][0]["sku_id"] == "SKU-1001"

        diagnosis = await client.call_tool(
            "diagnose_delivery_exception",
            {"order_id": "ORD-1002"},
        )
        assert diagnosis["recommended_action"] == "wait_for_next_scan"

        options = await client.call_tool(
            "evaluate_after_sales_options",
            {"order_id": "ORD-1001"},
        )
        assert "refund" in options["available_actions"]


def test_standard_mcp_initialize_list_and_call():
    run(protocol_scenario())


async def idempotent_refund_scenario():
    async with server_client() as client:
        params = {
            "order_id": "ORD-1001",
            "action_id": "action-refund-001",
            "reason": "quality_issue",
        }
        first = await client.call_tool("create_refund", params)
        second = await client.call_tool("create_refund", params)
        conflict = await client.call_tool(
            "create_refund",
            {
                **params,
                "order_id": "ORD-1002",
            },
        )

        assert first["created"] is True
        assert first["idempotent_replay"] is False
        assert second["created"] is True
        assert second["idempotent_replay"] is True
        assert second["refund_id"] == first["refund_id"]
        assert conflict["created"] is False
        assert conflict["error"] == "idempotency_conflict"


def test_create_refund_is_idempotent_by_action_id():
    run(idempotent_refund_scenario())


def test_refund_eligibility_uses_injected_business_clock(monkeypatch):
    monkeypatch.setitem(
        customer_service_server._ORDERS["ORD-1001"],
        "refundable_until",
        "2026-08-08",
    )
    monkeypatch.setattr(
        customer_service_server,
        "_current_date",
        lambda: date(2026, 8, 8),
        raising=False,
    )
    on_deadline = customer_service_server.check_refund_eligibility("ORD-1001")

    monkeypatch.setattr(
        customer_service_server,
        "_current_date",
        lambda: date(2026, 8, 9),
        raising=False,
    )
    after_deadline = customer_service_server.check_refund_eligibility("ORD-1001")

    assert on_deadline["eligible"] is True
    assert after_deadline["eligible"] is False


async def adapter_scenario():
    async with server_client() as client:
        adapters = await MCPToolAdapter.discover(
            client,
            write_tools={
                "create_refund",
                "create_return",
                "cancel_order",
                "create_ticket",
            },
        )
        by_name = {adapter.spec.name: adapter for adapter in adapters}

        assert by_name["query_order"].spec.tool_type == ToolType.READ
        assert by_name["create_refund"].spec.tool_type == ToolType.WRITE
        assert by_name["create_return"].spec.tool_type == ToolType.WRITE
        assert by_name["cancel_order"].spec.tool_type == ToolType.WRITE
        assert "order_id" in by_name["query_order"].spec.input_schema["required"]

        registry = ToolRegistry()
        for adapter in adapters:
            registry.register(adapter)
        registry.set_agent_whitelist(
            "after_sales",
            {"query_order", "check_refund_eligibility", "create_refund"},
        )

        order = await registry.call(
            "after_sales",
            "query_order",
            {"order_id": "ORD-1001"},
        )
        assert order.success
        assert order.data["found"] is True

        blocked = await registry.call(
            "after_sales",
            "create_refund",
            {
                "order_id": "ORD-1001",
                "action_id": "action-adapter-001",
                "reason": "quality_issue",
            },
            action_id="action-adapter-001",
        )
        assert not blocked.success
        assert "not confirmed" in blocked.error

        registry.confirm_action("action-adapter-001")
        created = await registry.call(
            "after_sales",
            "create_refund",
            {
                "order_id": "ORD-1001",
                "action_id": "action-adapter-001",
                "reason": "quality_issue",
            },
            action_id="action-adapter-001",
        )
        assert created.success
        assert created.data["created"] is True
        stats = registry.get_stats()
        assert stats["query_order"]["total"] == 1
        assert stats["create_refund"]["total"] == 1
        assert stats["create_refund"]["success_rate"] == 1.0


def test_mcp_tool_adapter_discovers_and_registers_tools():
    run(adapter_scenario())
