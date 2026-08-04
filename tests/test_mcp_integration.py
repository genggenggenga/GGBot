import asyncio
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters

from core.mcp_adapter import MCPClient, MCPToolAdapter
from core.tool_registry import ToolRegistry, ToolType


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
            "track_package",
            "check_refund_eligibility",
            "create_refund",
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

        eligibility = await client.call_tool(
            "check_refund_eligibility",
            {"order_id": "ORD-1001", "reason": "quality_issue"},
        )
        assert eligibility["eligible"] is True


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

        assert first["created"] is True
        assert first["idempotent_replay"] is False
        assert second["created"] is True
        assert second["idempotent_replay"] is True
        assert second["refund_id"] == first["refund_id"]


def test_create_refund_is_idempotent_by_action_id():
    run(idempotent_refund_scenario())


async def adapter_scenario():
    async with server_client() as client:
        adapters = await MCPToolAdapter.discover(
            client,
            write_tools={"create_refund", "create_ticket"},
        )
        by_name = {adapter.spec.name: adapter for adapter in adapters}

        assert by_name["query_order"].spec.tool_type == ToolType.READ
        assert by_name["create_refund"].spec.tool_type == ToolType.WRITE
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


def test_mcp_tool_adapter_discovers_and_registers_tools():
    run(adapter_scenario())
