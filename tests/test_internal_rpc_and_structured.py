from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

from agents.domain_agents import AfterSalesAgent, OrderAgent
from core.agent_models import AgentDecision, DecisionType, DialogueState
from core.idempotency import (
    IdempotencyConflictError,
    InMemoryActionExecutionRepository,
    RedisActionExecutionRepository,
)
from core.internal_rpc import build_mock_rpc_clients
from core.rpc_tools import register_internal_rpc_tools
from core.skill_loader import SkillManager
from core.structured_llm import StructuredLLMClient
from core.tool_names import canonical_tool_name
from core.tool_registry import ToolRegistry
from core.prompts.types import PromptSpec


@pytest.mark.asyncio
async def test_internal_rpc_tools_use_shared_idempotency_repository():
    registry = ToolRegistry()
    register_internal_rpc_tools(
        registry,
        build_mock_rpc_clients(),
        InMemoryActionExecutionRepository(),
    )
    query_tool = canonical_tool_name("query_order")
    refund_tool = canonical_tool_name("create_refund")
    registry.set_agent_whitelist("order", {query_tool})
    registry.set_agent_whitelist("after_sales", {refund_tool})

    order = await registry.call(
        "order",
        query_tool,
        {"order_id": "ORD-1001"},
    )
    assert order.success
    assert order.data["status"] == "delivered"

    params = {"order_id": "ORD-1001", "action_id": "action-1"}
    registry.confirm_action("action-1")
    first = await registry.call(
        "after_sales",
        refund_tool,
        params,
        action_id="action-1",
    )
    registry.confirm_action("action-1")
    replay = await registry.call(
        "after_sales",
        refund_tool,
        params,
        action_id="action-1",
    )

    assert first.success and replay.success
    assert first.data["refund_id"] == replay.data["refund_id"]
    assert replay.data["idempotent_replay"] is True

    registry.confirm_action("action-1")
    conflict = await registry.call(
        "after_sales",
        refund_tool,
        {"order_id": "ORD-1002", "action_id": "action-1"},
        action_id="action-1",
    )
    assert not conflict.success
    assert "idempotency_conflict" in conflict.error


class FakeHashRedis:
    def __init__(self):
        self.hashes = {}

    async def eval(self, script, key_count, key, *args):
        assert key_count == 1
        if 'redis.call("exists"' in script:
            if key in self.hashes:
                return 0
            fingerprint, owner_token, _ttl = args
            self.hashes[key] = {
                "fingerprint": fingerprint,
                "status": "processing",
                "owner_token": owner_token,
                "result": "",
            }
            return 1

        owner_token, status, result, _ttl = args
        record = self.hashes.get(key)
        if (
            record is None
            or record["owner_token"] != owner_token
            or record["status"] != "processing"
        ):
            return 0
        record.update(status=status, result=result)
        return 1

    async def hgetall(self, key):
        return dict(self.hashes.get(key, {}))


@pytest.mark.asyncio
async def test_redis_idempotency_is_shared_between_repository_instances():
    redis = FakeHashRedis()
    first_repo = RedisActionExecutionRepository(redis)
    second_repo = RedisActionExecutionRepository(redis)
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        return {"created": True, "refund_id": "REF-1"}

    first = await first_repo.execute(
        "create_refund",
        "action-1",
        {"order_id": "ORD-1001"},
        operation,
    )
    replay = await second_repo.execute(
        "create_refund",
        "action-1",
        {"order_id": "ORD-1001"},
        operation,
    )

    assert first["refund_id"] == replay["refund_id"]
    assert replay["idempotent_replay"] is True
    assert calls == 1

    with pytest.raises(IdempotencyConflictError):
        await second_repo.execute(
            "create_refund",
            "action-1",
            {"order_id": "ORD-1002"},
            operation,
        )


class StructuredResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str


@pytest.mark.asyncio
async def test_structured_llm_forces_tool_call_and_validates_schema():
    calls = []

    class Messages:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(content=[
                SimpleNamespace(
                    type="tool_use",
                    name="submit_result",
                    input={"value": "ok"},
                ),
            ])

    client = StructuredLLMClient(
        SimpleNamespace(messages=Messages()),
        "test-model",
    )
    result = await client.generate(
        PromptSpec(system="system", user="user"),
        StructuredResult,
        tool_name="submit_result",
    )

    assert result.value == "ok"
    assert calls[0]["tool_choice"] == {
        "type": "tool",
        "name": "submit_result",
    }
    assert calls[0]["tools"][0]["input_schema"]["additionalProperties"] is False


class RecordingPlanner:
    def __init__(self):
        self.calls = []

    async def decide(self, **kwargs):
        self.calls.append(kwargs)
        return AgentDecision(
            type=DecisionType.FINISH,
            response="已按售后流程处理。",
            reason_code="done",
        )


@pytest.mark.asyncio
async def test_only_after_sales_agent_resolves_skill_context():
    registry = ToolRegistry()
    register_internal_rpc_tools(
        registry,
        build_mock_rpc_clients(),
        InMemoryActionExecutionRepository(),
    )
    skills = SkillManager(str(Path(__file__).parent.parent / "skills"))
    skills.load()
    after_sales_planner = RecordingPlanner()
    order_planner = RecordingPlanner()
    after_sales = AfterSalesAgent(
        registry,
        planner=after_sales_planner,
        skill_manager=skills,
    )
    order = OrderAgent(registry, planner=order_planner)

    await after_sales.execute(
        DialogueState(
            active_intent="refund_request",
            slots={"order_id": "ORD-1001"},
        ),
        "我要退款",
    )
    await order.execute(
        DialogueState(
            active_intent="order_query",
            slots={"order_id": "ORD-1001"},
        ),
        "查询订单",
    )

    assert "售后通用安全基线" in after_sales_planner.calls[0]["skill_context"]
    assert "退款申请 SOP" in after_sales_planner.calls[0]["skill_context"]
    assert "退货申请 SOP" not in after_sales_planner.calls[0]["skill_context"]
    assert order_planner.calls[0]["skill_context"] == ""
