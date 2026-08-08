import json

import pytest

from agents.domain_agents import AfterSalesAgent, LogisticsAgent, OrderAgent
from core.agent_models import (
    AgentDecision,
    ConfirmationStatus,
    DecisionType,
    DialogueState,
)
from core.react_planner import ReActPlanner
from core.tool_registry import (
    LocalToolAdapter,
    ToolRegistry,
    ToolSpec,
    ToolType,
)


def register_tool(
    registry,
    name,
    handler,
    *,
    required=(),
    tool_type=ToolType.READ,
):
    registry.register(LocalToolAdapter(
        ToolSpec(
            name=name,
            description=f"test {name}",
            input_schema={
                "type": "object",
                "properties": {
                    field: {"type": "string"}
                    for field in required
                },
                "required": list(required),
            },
            tool_type=tool_type,
        ),
        handler,
    ))


class SequencePlanner:
    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.calls = []

    async def decide(self, **kwargs):
        self.calls.append(kwargs)
        decision = self.decisions.pop(0)
        if isinstance(decision, Exception):
            raise decision
        return decision


@pytest.mark.asyncio
async def test_react_planner_parses_structured_decision():
    prompts = []

    async def llm_call(prompt):
        prompts.append(prompt)
        return json.dumps({
            "type": "tool",
            "tool_name": "query_order",
            "arguments": {"order_id": "ORD-1"},
            "response": None,
            "reason_code": "need_order_fact",
        })

    planner = ReActPlanner(llm_call)
    decision = await planner.decide(
        agent_name="order",
        goal="order_query",
        message="查订单",
        state=DialogueState(
            active_intent="order_query",
            slots={"order_id": "ORD-1"},
        ),
        observations=[],
        tools=[ToolSpec(
            name="query_order",
            description="query",
            input_schema={"type": "object"},
        )],
    )

    assert decision.type == DecisionType.TOOL
    assert decision.tool_name == "query_order"
    assert "不输出 Thought" in prompts[0]


@pytest.mark.asyncio
async def test_logistics_react_selects_tool_from_observation_loop():
    registry = ToolRegistry()
    tool_calls = []

    async def diagnose(params, context):
        del context
        tool_calls.append(params)
        return {
            "found": True,
            "exception_type": "in_transit",
            "recommended_action": "wait_for_next_scan",
        }

    register_tool(
        registry,
        "diagnose_delivery_exception",
        diagnose,
        required=("order_id",),
    )
    planner = SequencePlanner(
        AgentDecision(
            type=DecisionType.TOOL,
            tool_name="diagnose_delivery_exception",
            arguments={"order_id": "ORD-1"},
            reason_code="diagnose_delay",
        ),
        AgentDecision(
            type=DecisionType.FINISH,
            response="包裹仍在运输中，建议等待下一次物流扫描。",
            reason_code="diagnosis_complete",
        ),
    )
    agent = LogisticsAgent(registry, planner=planner)

    result = await agent.execute(DialogueState(
        active_intent="logistics_query",
        slots={"order_id": "ORD-1"},
    ))

    assert result.success
    assert tool_calls == [{"order_id": "ORD-1"}]
    assert result.observations[0].name == "diagnose_delivery_exception"
    assert len(planner.calls[1]["observations"]) == 1


@pytest.mark.asyncio
async def test_react_write_tool_becomes_pending_action_before_execution():
    registry = ToolRegistry()
    writes = []

    async def create_refund(params, context):
        del context
        writes.append(params)
        return {
            "created": True,
            "refund_id": "REF-1",
            **params,
        }

    register_tool(
        registry,
        "create_refund",
        create_refund,
        required=("order_id", "action_id"),
        tool_type=ToolType.WRITE,
    )
    planner = SequencePlanner(
        AgentDecision(
            type=DecisionType.TOOL,
            tool_name="create_refund",
            arguments={"order_id": "ORD-1"},
            response="订单符合退款条件，请确认提交。",
            reason_code="refund_ready",
        ),
        AgentDecision(
            type=DecisionType.FINISH,
            response="退款申请已提交，申请编号 REF-1。",
            reason_code="refund_created",
        ),
    )
    agent = AfterSalesAgent(registry, planner=planner)
    state = DialogueState(
        active_intent="refund_request",
        slots={"order_id": "ORD-1"},
    )

    pending = await agent.execute(state)

    assert pending.pending_action is not None
    assert writes == []

    confirmed = state.model_copy(update={
        "pending_action": pending.pending_action,
        "confirmation_status": ConfirmationStatus.CONFIRMED,
    })
    completed = await agent.execute(confirmed)

    assert completed.success
    assert len(writes) == 1
    assert writes[0]["action_id"] == pending.pending_action.action_id
    assert "REF-1" in completed.response


@pytest.mark.asyncio
async def test_react_rejects_tool_outside_agent_whitelist():
    registry = ToolRegistry()

    async def create_refund(params, context):
        return params

    register_tool(
        registry,
        "create_refund",
        create_refund,
        tool_type=ToolType.WRITE,
    )
    planner = SequencePlanner(AgentDecision(
        type=DecisionType.TOOL,
        tool_name="create_refund",
        arguments={"order_id": "ORD-1"},
        reason_code="unauthorized_attempt",
    ))

    result = await OrderAgent(registry, planner=planner).execute(
        DialogueState(
            active_intent="order_query",
            slots={"order_id": "ORD-1"},
        ),
    )

    assert not result.success
    assert result.error == "react_tool_not_allowed:create_refund"


@pytest.mark.asyncio
async def test_react_planner_failure_uses_deterministic_fallback():
    registry = ToolRegistry()

    async def query_order(params, context):
        return {"found": True, "status": "paid", **params}

    register_tool(
        registry,
        "query_order",
        query_order,
        required=("order_id",),
    )
    planner = SequencePlanner(RuntimeError("LLM unavailable"))

    result = await OrderAgent(registry, planner=planner).execute(
        DialogueState(
            active_intent="order_query",
            slots={"order_id": "ORD-1"},
        ),
    )

    assert result.success
    assert result.observations[0].name == "query_order"
    assert "paid" in result.response
