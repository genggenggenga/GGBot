import asyncio
from pathlib import Path

import pytest

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
    PendingAction,
)
from core.skill_loader import SkillManager
from core.tool_registry import (
    LocalToolAdapter,
    ToolRegistry,
    ToolSpec,
    ToolType,
)
from rag.query_planner import QueryPlanner


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
    assert router.route(DialogueState(active_intent="request")) == AFTER_SALES_AGENT
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
    assert "query_payment" not in agent.allowed_tools


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


def test_logistics_agent_can_track_by_tracking_number_directly():
    registry = ToolRegistry()
    calls = []

    async def track_package(params, context):
        calls.append(dict(params))
        return {"found": True, "status": "in_transit", **params}

    register_tool(
        registry,
        "track_package",
        track_package,
        required=("tracking_no",),
    )
    result = run(LogisticsAgent(registry).execute(DialogueState(
        active_intent="logistics_query",
        slots={"tracking_no": "SF1234567890"},
        required_slots=["order_id", "tracking_no"],
    )))

    assert result.success
    assert calls == [{"tracking_no": "SF1234567890"}]


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
        return {"created": True, "refund_id": "REF-1", **params}

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


@pytest.mark.parametrize(
    "intent",
    ["return_request", "cancel_order", "request"],
)
def test_after_sales_fails_closed_for_unimplemented_intents(intent):
    registry = ToolRegistry()
    calls = []

    async def query_order(params, context):
        calls.append(("query_order", params))
        return {"found": True, "order_id": params["order_id"]}

    async def check_eligibility(params, context):
        calls.append(("check_refund_eligibility", params))
        return {"eligible": True, "order_id": params["order_id"]}

    register_tool(registry, "query_order", query_order)
    register_tool(registry, "check_refund_eligibility", check_eligibility)
    result = run(AfterSalesAgent(registry).execute(DialogueState(
        active_intent=intent,
        slots={"order_id": "ORD-1"} if "request" in intent or intent == "cancel_order" else {},
    )))

    assert result.success
    assert result.completed is False
    assert "尚未实现" in result.response
    assert calls == []


@pytest.mark.parametrize("intent", ["complaint", "escalation"])
def test_after_sales_creates_handoff_ticket_after_confirmation(intent):
    registry = ToolRegistry()
    create_calls = []

    async def create_ticket(params, context):
        create_calls.append(dict(params))
        return {
            "created": True,
            "ticket_id": "TKT-1001",
            **params,
        }

    register_tool(
        registry,
        "create_ticket",
        create_ticket,
        required=("subject", "description", "action_id"),
        tool_type=ToolType.WRITE,
    )
    agent = AfterSalesAgent(registry)
    state = DialogueState(active_intent=intent)

    pending = run(agent.execute(state, "用户要求转人工处理"))
    assert pending.pending_action is not None
    assert pending.pending_action.tool_name == "create_ticket"
    assert create_calls == []

    confirmed = state.model_copy(update={
        "pending_action": pending.pending_action,
        "confirmation_status": ConfirmationStatus.CONFIRMED,
    })
    completed = run(agent.execute(confirmed, "确认"))

    assert completed.success
    assert completed.completed
    assert "TKT-1001" in completed.response
    assert len(create_calls) == 1


def test_order_agent_handles_business_not_found_without_none_status():
    registry = ToolRegistry()

    async def query_order(params, context):
        return {
            "found": False,
            "order_id": params["order_id"],
            "error": "order_not_found",
        }

    register_tool(registry, "query_order", query_order)
    result = run(OrderAgent(registry).execute(DialogueState(
        active_intent="order_query",
        slots={"order_id": "ORD-MISSING"},
        required_slots=["order_id"],
    )))

    assert result.success
    assert "没有找到" in result.response
    assert "None" not in result.response


def test_logistics_agent_handles_missing_tracking_without_none_status():
    registry = ToolRegistry()

    async def query_order(params, context):
        return {"found": True, "order_id": params["order_id"], "status": "paid"}

    async def track_package(params, context):
        return {
            "found": False,
            "order_id": params["order_id"],
            "error": "tracking_not_available",
        }

    register_tool(registry, "query_order", query_order)
    register_tool(registry, "track_package", track_package)
    result = run(LogisticsAgent(registry).execute(DialogueState(
        active_intent="logistics_query",
        slots={"order_id": "ORD-1"},
        required_slots=["order_id"],
    )))

    assert result.success
    assert "暂无物流" in result.response
    assert "None" not in result.response


def test_after_sales_handles_rejected_refund_creation_as_business_result():
    registry = ToolRegistry()

    async def create_refund(params, context):
        return {
            "created": False,
            "order_id": params["order_id"],
            "action_id": params["action_id"],
            "error": "outside_refund_window",
        }

    register_tool(
        registry,
        "create_refund",
        create_refund,
        required=("order_id", "action_id"),
        tool_type=ToolType.WRITE,
    )
    agent = AfterSalesAgent(registry)
    pending = PendingAction(
        tool_name="create_refund",
        arguments={"order_id": "ORD-1"},
    )
    state = DialogueState(
        active_intent="refund_request",
        slots={"order_id": "ORD-1"},
        pending_action=pending,
        confirmation_status=ConfirmationStatus.CONFIRMED,
    )

    result = run(agent.execute(state))

    assert result.success
    assert "未能提交" in result.response
    assert "None" not in result.response


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


def test_knowledge_agent_uses_skill_and_memory_context_in_retrieval_query():
    registry = ToolRegistry()
    queries = []

    async def rag_search(params, context):
        queries.append(params["query"])
        return {
            "answered": True,
            "hits": [{"chunk": {"content": "上下文相关答案"}}],
            "citations": [{"citation_id": "[1]", "source": "policy.md"}],
        }

    register_tool(registry, "rag_search", rag_search, required=("query",))
    context = (
        "[Skills]\n退款需要先核验订单。\n\n"
        "[会话摘要]\n用户此前询问退款到账。\n\n"
        "[相关历史]\n- 上次退款使用原支付渠道。\n\n"
        "[用户画像]\n{\"language\":\"zh\"}"
    )
    result = run(KnowledgeAgent(registry).execute(
        DialogueState(active_intent="refund_policy"),
        "这个要多久",
        context,
    ))

    assert result.success
    assert len(queries) == 1
    assert "这个要多久" in queries[0]
    assert "退款需要先核验订单" in queries[0]
    assert "用户此前询问退款到账" in queries[0]
    assert "上次退款使用原支付渠道" in queries[0]
    assert "language" in queries[0]


def test_knowledge_agent_sends_bounded_multi_query_plan_to_rag():
    registry = ToolRegistry()
    params_seen = []

    async def rag_search(params, context):
        params_seen.append(params)
        return {
            "answered": True,
            "hits": [{"chunk": {"content": "退款到账需要五个工作日"}}],
            "citations": [{"citation_id": "[1]", "source": "policy.md"}],
        }

    async def llm_call(prompt):
        return """{
          "standalone_query": "退款审核通过后多久到账",
          "alternative_queries": ["退款到账时间", "退款原路退回周期"],
          "resolved_references": {"它": "退款款项"},
          "confidence": 0.95
        }"""

    register_tool(registry, "rag_search", rag_search, required=("query",))
    planner = QueryPlanner(llm_call, max_queries=3)
    result = run(KnowledgeAgent(registry, planner).execute(
        DialogueState(active_intent="refund_policy"),
        "它多久到账",
        history=[{"role": "user", "content": "退款已经审核通过"}],
    ))

    assert result.success
    assert params_seen[0]["query"] == "退款审核通过后多久到账"
    assert params_seen[0]["queries"] == [
        "退款审核通过后多久到账",
        "它多久到账",
        "退款到账时间",
    ]


@pytest.mark.parametrize(
    ("intent", "expected_text"),
    [
        ("greeting", "你好"),
        ("feedback", "感谢"),
        ("other", "订单、物流"),
    ],
)
def test_knowledge_agent_uses_deterministic_responses_without_rag(
    intent,
    expected_text,
):
    registry = ToolRegistry()
    calls = 0

    async def rag_search(params, context):
        nonlocal calls
        calls += 1
        return {"answered": True, "hits": [{"content": "不应返回"}]}

    register_tool(registry, "rag_search", rag_search, required=("query",))
    result = run(KnowledgeAgent(registry).execute(
        DialogueState(active_intent=intent),
        "测试消息",
    ))

    assert result.success
    assert expected_text in result.response
    assert calls == 0


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

        async def execute(self, state, message, **kwargs):
            events.append(self.name)
            return type("Result", (), {
                "response": self.name,
                "success": True,
                "completed": True,
                "observations": [],
                "pending_action": None,
                "citations": [],
            })()

    runtime = DomainAgentRuntime(
        Router(),
        {
            ORDER_AGENT: FakeAgent(ORDER_AGENT),
            LOGISTICS_AGENT: FakeAgent(LOGISTICS_AGENT),
        },
    )
    response, _ = run(runtime.execute(
        DialogueState(
            active_intent="order_query",
            slots={"order_id": "ORD-1"},
        ),
        "查订单和物流",
        ["order_query", "logistics_query"],
    ))

    assert events == [ORDER_AGENT, LOGISTICS_AGENT]
    assert response == "order\n\nlogistics"


def test_runtime_evaluates_completion_condition_for_each_goal():
    registry = ToolRegistry()

    async def query_order(params, context):
        return {
            "found": True,
            "order_id": params["order_id"],
            "status": "paid",
        }

    async def track_package(params, context):
        return {
            "found": True,
            "order_id": params["order_id"],
            "status": "in_transit",
        }

    register_tool(registry, "query_order", query_order)
    register_tool(registry, "track_package", track_package)
    runtime = DomainAgentRuntime(
        Router(),
        {
            ORDER_AGENT: OrderAgent(registry),
            LOGISTICS_AGENT: LogisticsAgent(registry),
        },
    )

    _, results = run(runtime.execute(
        DialogueState(
            active_intent="order_query",
            slots={"order_id": "ORD-1"},
        ),
        "查订单和物流",
        ["order_query", "logistics_query"],
    ))

    assert [result.goal for result in results] == [
        "order_query",
        "logistics_query",
    ]
    assert all(result.completed for result in results)


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
