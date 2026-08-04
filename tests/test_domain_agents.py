import asyncio
from pathlib import Path

from agents.domain_agents import (
    AFTER_SALES_AGENT,
    KNOWLEDGE_AGENT,
    LOGISTICS_AGENT,
    ORDER_AGENT,
    AfterSalesAgent,
    DomainAgentRuntime,
    KnowledgeAgent,
    LogisticsAgent,
    OrderAgent,
    Router,
    ServiceAgent,
)
from core.agent_models import (
    ConfirmationStatus,
    DialogueState,
    Observation,
)
from core.skill_loader import SkillManager
from core.tool_registry import (
    LocalToolAdapter,
    ToolRegistry,
    ToolSpec,
    ToolType,
)


def run(coro):
    return asyncio.run(coro)


def register_tool(
    registry,
    name,
    handler,
    *,
    required=("order_id",),
    tool_type=ToolType.READ,
):
    properties = {field: {"type": "string"} for field in required}
    registry.register(LocalToolAdapter(
        ToolSpec(
            name=name,
            description=f"test {name}",
            input_schema={
                "type": "object",
                "properties": properties,
                "required": list(required),
            },
            tool_type=tool_type,
        ),
        handler,
    ))


def test_router_uses_dialogue_state_and_deduplicates_tasks():
    router = Router()

    assert router.route(DialogueState(active_intent="refund_policy")) == KNOWLEDGE_AGENT
    assert router.route(DialogueState(active_intent="order_query")) == ORDER_AGENT
    assert router.route(DialogueState(active_intent="logistics_query")) == LOGISTICS_AGENT
    assert router.route(DialogueState(active_intent="refund_request")) == AFTER_SALES_AGENT
    assert router.route_tasks(
        DialogueState(active_intent="order_query"),
        ["order_query", "logistics_query", "order_query"],
    ) == [ORDER_AGENT, LOGISTICS_AGENT]


def test_order_agent_uses_tool_observation():
    registry = ToolRegistry()

    async def query_order(params, context):
        return {"order_id": params["order_id"], "status": "paid"}

    register_tool(registry, "query_order", query_order)
    agent = OrderAgent(registry)
    result = run(agent.execute(DialogueState(
        active_intent="order_query",
        slots={"order_id": "ORD-1"},
        required_slots=["order_id"],
    )))

    assert result.success
    assert result.observations[0].name == "query_order"
    assert "paid" in result.response


def test_logistics_agent_runs_bounded_plan_in_order():
    registry = ToolRegistry()
    calls = []

    async def query_order(params, context):
        calls.append("query_order")
        return {"order_id": params["order_id"], "status": "shipped"}

    async def track_package(params, context):
        calls.append("track_package")
        return {"order_id": params["order_id"], "status": "in_transit"}

    register_tool(registry, "query_order", query_order)
    register_tool(registry, "track_package", track_package)
    result = run(LogisticsAgent(registry).execute(DialogueState(
        active_intent="logistics_query",
        slots={"order_id": "ORD-1"},
        required_slots=["order_id"],
    )))

    assert result.success
    assert calls == ["query_order", "track_package"]
    assert [item.name for item in result.observations] == calls


def test_service_agent_stops_plan_over_max_steps():
    class LongPlanAgent(ServiceAgent):
        name = "long"
        allowed_tools = ("unused",)

        def next_action(self, state, message, observations):
            return "unused", {}, None

        def finish(self, state, observations):
            raise AssertionError("must not finish")

    registry = ToolRegistry()

    async def unused(params, context):
        return {}

    register_tool(registry, "unused", unused, required=())
    result = run(LongPlanAgent(registry, max_steps=1).execute(DialogueState()))

    assert not result.success
    assert result.error == "max_steps_exceeded"


def test_after_sales_creates_pending_action_then_executes_after_confirmation():
    registry = ToolRegistry()
    create_calls = 0

    async def query_order(params, context):
        return {"order_id": params["order_id"], "status": "delivered"}

    async def check_eligibility(params, context):
        return {"order_id": params["order_id"], "eligible": True}

    async def create_refund(params, context):
        nonlocal create_calls
        create_calls += 1
        return {"refund_id": "REF-1", **params}

    register_tool(registry, "query_order", query_order)
    register_tool(
        registry,
        "check_refund_eligibility",
        check_eligibility,
    )
    register_tool(
        registry,
        "create_refund",
        create_refund,
        required=("order_id", "action_id"),
        tool_type=ToolType.WRITE,
    )
    agent = AfterSalesAgent(registry)
    state = DialogueState(
        active_intent="refund_request",
        slots={"order_id": "ORD-1"},
        required_slots=["order_id"],
    )

    pending = run(agent.execute(state))
    assert pending.pending_action is not None
    assert pending.completed is False
    assert create_calls == 0

    confirmed = state.model_copy(update={
        "pending_action": pending.pending_action,
        "confirmation_status": ConfirmationStatus.CONFIRMED,
    })
    completed = run(agent.execute(confirmed))

    assert completed.success
    assert "REF-1" in completed.response
    assert create_calls == 1


def test_knowledge_agent_calls_rag_once_and_returns_citation():
    registry = ToolRegistry()
    calls = 0

    async def rag_search(params, context):
        nonlocal calls
        calls += 1
        return {
            "answered": True,
            "hits": [{
                "chunk": {"content": "购买后七天内可以申请退款。"},
                "score": 0.9,
            }],
            "citations": [{"citation_id": "[1]", "source": "policy.md"}],
        }

    register_tool(registry, "rag_search", rag_search, required=("query",))
    result = run(KnowledgeAgent(registry).execute(
        DialogueState(active_intent="refund_policy"),
        "退款期限是什么",
    ))

    assert calls == 1
    assert result.success
    assert result.response.endswith("[1]")
    assert result.citations[0]["source"] == "policy.md"


def test_tool_registry_rejects_domain_agent_overreach():
    registry = ToolRegistry()

    async def create_refund(params, context):
        return params

    register_tool(
        registry,
        "create_refund",
        create_refund,
        required=("order_id",),
        tool_type=ToolType.WRITE,
    )
    OrderAgent(registry)

    result = run(registry.call(
        ORDER_AGENT,
        "create_refund",
        {"order_id": "ORD-1"},
        action_id="A-1",
    ))

    assert not result.success
    assert "not allowed" in result.error


def test_runtime_executes_composite_tasks_sequentially():
    events = []

    class FakeAgent:
        def __init__(self, name):
            self.name = name

        async def execute(self, state, message):
            events.append(self.name)
            return type("Result", (), {
                "response": self.name,
            })()

    runtime = DomainAgentRuntime(
        Router(),
        {
            ORDER_AGENT: FakeAgent(ORDER_AGENT),
            LOGISTICS_AGENT: FakeAgent(LOGISTICS_AGENT),
        },
    )
    response, _ = run(runtime.execute(
        DialogueState(active_intent="order_query"),
        "查订单和物流",
        ["order_query", "logistics_query"],
    ))

    assert events == [ORDER_AGENT, LOGISTICS_AGENT]
    assert response == "order\n\nlogistics"


def test_skill_loader_maps_legacy_agents_and_reads_new_metadata():
    root = Path(__file__).parent.parent / "skills"
    manager = SkillManager(str(root))
    skills = manager.load()
    billing = next(skill for skill in skills if skill.name == "账单退款处理规范")

    assert billing.agents == [KNOWLEDGE_AGENT, AFTER_SALES_AGENT]
    assert "refund_request" in billing.intents
    assert billing.version == "2"
    assert billing.eval_cases == ["refund_policy", "refund_request"]
    assert billing.matches("我要退款", AFTER_SALES_AGENT, "refund_request")
