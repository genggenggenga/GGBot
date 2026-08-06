import asyncio

import pytest

from core.tool_registry import (
    LocalToolAdapter,
    ToolRegistry,
    ToolSpec,
    ToolType,
)


def run(coro):
    return asyncio.run(coro)


def make_spec(
    name: str = "query_order",
    *,
    tool_type: ToolType = ToolType.READ,
    timeout_s: float = 0.1,
    cache_ttl: float = 0,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="test tool",
        input_schema={
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "mode": {"type": "string", "enum": ["brief", "full"]},
            },
            "required": ["order_id"],
        },
        tool_type=tool_type,
        timeout_s=timeout_s,
        cache_ttl=cache_ttl,
    )


def test_tool_spec_rejects_invalid_timeout():
    with pytest.raises(ValueError):
        make_spec(timeout_s=0)


def test_registry_rejects_missing_and_invalid_parameters():
    async def handler(params, context):
        return params

    registry = ToolRegistry()
    registry.register(LocalToolAdapter(make_spec(), handler))
    registry.set_agent_whitelist("order", {"query_order"})

    missing = run(registry.call("order", "query_order", {}))
    wrong_type = run(registry.call("order", "query_order", {"order_id": 123}))
    wrong_enum = run(
        registry.call(
            "order",
            "query_order",
            {"order_id": "O-1", "mode": "invalid"},
        )
    )

    assert not missing.success
    assert "missing required parameter" in missing.error
    assert not wrong_type.success
    assert "type error" in wrong_type.error
    assert not wrong_enum.success
    assert "must be one of" in wrong_enum.error


def test_registry_rejects_agent_without_permission():
    called = 0

    async def handler(params, context):
        nonlocal called
        called += 1
        return {"id": params["order_id"]}

    registry = ToolRegistry()
    registry.register(LocalToolAdapter(make_spec(), handler))

    result = run(
        registry.call("knowledge", "query_order", {"order_id": "O-1"})
    )

    assert not result.success
    assert "not allowed" in result.error
    assert called == 0


def test_local_adapter_timeout_uses_fallback():
    async def handler(params, context):
        await asyncio.sleep(0.05)
        return {"unexpected": True}

    def fallback(params, context, error):
        return {"fallback": True, "reason": error}

    registry = ToolRegistry()
    spec = make_spec(timeout_s=0.001)
    registry.register(LocalToolAdapter(spec, handler, fallback=fallback))
    registry.set_agent_whitelist("order", {"query_order"})

    result = run(
        registry.call("order", "query_order", {"order_id": "O-1"})
    )

    assert result.success
    assert result.data["fallback"] is True
    assert result.error == "execution timeout"


def test_local_adapter_cache_avoids_repeated_handler_call():
    called = 0

    async def handler(params, context):
        nonlocal called
        called += 1
        return {"id": params["order_id"]}

    registry = ToolRegistry()
    registry.register(
        LocalToolAdapter(make_spec(cache_ttl=60), handler)
    )
    registry.set_agent_whitelist("order", {"query_order"})

    first = run(
        registry.call("order", "query_order", {"order_id": "O-1"})
    )
    second = run(
        registry.call("order", "query_order", {"order_id": "O-1"})
    )

    assert first.success and not first.cached
    assert second.success and second.cached
    assert called == 1


def test_write_tool_requires_explicit_confirmation():
    called = 0

    async def handler(params, context):
        nonlocal called
        called += 1
        return {"refund_id": "R-1"}

    registry = ToolRegistry()
    registry.register(
        LocalToolAdapter(
            make_spec(name="create_refund", tool_type=ToolType.WRITE),
            handler,
        )
    )
    registry.set_agent_whitelist("after_sales", {"create_refund"})

    missing_action = run(
        registry.call(
            "after_sales",
            "create_refund",
            {"order_id": "O-1"},
        )
    )
    pending = run(
        registry.call(
            "after_sales",
            "create_refund",
            {"order_id": "O-1"},
            action_id="A-1",
        )
    )

    assert not missing_action.success
    assert "requires action_id" in missing_action.error
    assert not pending.success
    assert "not confirmed" in pending.error
    assert called == 0

    registry.confirm_action("A-1")
    confirmed = run(
        registry.call(
            "after_sales",
            "create_refund",
            {"order_id": "O-1"},
            action_id="A-1",
        )
    )

    assert confirmed.success
    assert confirmed.data == {"refund_id": "R-1"}
    assert called == 1
    assert registry.is_action_confirmed("A-1") is False

    replay_without_confirmation = run(
        registry.call(
            "after_sales",
            "create_refund",
            {"order_id": "O-1"},
            action_id="A-1",
        )
    )
    assert not replay_without_confirmation.success


def test_tool_result_can_convert_to_observation():
    async def handler(params, context):
        return {"order_id": params["order_id"], "status": "paid"}

    registry = ToolRegistry()
    registry.register(LocalToolAdapter(make_spec(), handler))
    registry.set_agent_whitelist("order", {"query_order"})

    result = run(
        registry.call("order", "query_order", {"order_id": "O-1"})
    )
    observation = result.to_observation()

    assert observation.source == "tool"
    assert observation.name == "query_order"
    assert observation.success is True
    assert observation.data["status"] == "paid"


def test_legacy_tool_manager_remains_importable():
    from mcp.tool_manager import MCPToolManager, Tool

    assert MCPToolManager is not None
    assert Tool is not None
