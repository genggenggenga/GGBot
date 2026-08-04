import asyncio

from agents.domain_agents import (
    AfterSalesAgent,
    DomainAgentRuntime,
    KnowledgeAgent,
    LogisticsAgent,
    OrderAgent,
    Router,
)
from api import main as api_main
from api.main import ChatRequest, ChatResponse
from core.customer_agent_runtime import CustomerAgentRuntime
from core.customer_agent_runtime import CustomerTurnResult
from core.dialogue_state_tracker import DialogueStateTracker
from core.intent_recognizer import IntentRecognizer
from core.state_store import InMemoryStateStore
from core.tool_registry import (
    LocalToolAdapter,
    ToolRegistry,
    ToolSpec,
    ToolType,
)
from core.trace_store import TraceStore
from core.turn_engine import TurnEngine


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


def build_runtime(*, eligible=True, fail_query=False):
    store = InMemoryStateStore()
    registry = ToolRegistry()
    calls = []

    async def query_order(params, context):
        calls.append(("query_order", dict(params)))
        if fail_query:
            raise RuntimeError("order service unavailable")
        return {
            "found": params["order_id"] != "ORD-9999",
            "order_id": params["order_id"],
            "status": "delivered",
        }

    async def track_package(params, context):
        calls.append(("track_package", dict(params)))
        return {"found": True, "status": "in_transit", **params}

    async def check_refund(params, context):
        calls.append(("check_refund_eligibility", dict(params)))
        return {
            "eligible": eligible,
            "reason": "outside_refund_window" if not eligible else "within_window",
            **params,
        }

    async def create_refund(params, context):
        calls.append(("create_refund", dict(params)))
        return {"created": True, "refund_id": "REF-1001", **params}

    async def rag_search(params, context):
        calls.append(("rag_search", dict(params)))
        return {
            "answered": True,
            "hits": [{
                "chunk": {"content": "购买后七天内可以申请退款。"},
                "score": 0.9,
            }],
            "citations": [{"citation_id": "[1]", "source": "policy.md"}],
        }

    register_tool(registry, "query_order", query_order)
    register_tool(registry, "track_package", track_package)
    register_tool(registry, "check_refund_eligibility", check_refund)
    register_tool(
        registry,
        "create_refund",
        create_refund,
        required=("order_id", "action_id"),
        tool_type=ToolType.WRITE,
    )
    register_tool(registry, "rag_search", rag_search, required=("query",))

    router = Router()
    domain = DomainAgentRuntime(router, {
        "knowledge": KnowledgeAgent(registry),
        "order": OrderAgent(registry),
        "logistics": LogisticsAgent(registry),
        "after_sales": AfterSalesAgent(registry),
    })
    trace_store = TraceStore()
    runtime = CustomerAgentRuntime(
        recognizer=IntentRecognizer(api_key="test"),
        tracker=DialogueStateTracker(),
        turn_engine=TurnEngine(store),
        domain_runtime=domain,
        router=router,
        trace_store=trace_store,
    )
    return runtime, store, calls, trace_store


def test_refund_flow_clarifies_confirms_and_creates_once():
    runtime, store, calls, traces = build_runtime()

    first = run(runtime.run("user-1", "conv-1", "我要退款"))
    assert first.status == "awaiting_user"
    assert first.missing_slots == ["order_id"]
    assert calls == []

    second = run(runtime.run("user-1", "conv-1", "订单号 ORD-1001"))
    assert second.status == "awaiting_user"
    assert "确认" in second.response
    assert [name for name, _ in calls] == [
        "query_order",
        "check_refund_eligibility",
    ]
    assert all(name != "create_refund" for name, _ in calls)

    third = run(runtime.run("user-1", "conv-1", "确认"))
    assert third.status == "completed"
    assert third.agent_type == "after_sales"
    assert "REF-1001" in third.response
    assert [name for name, _ in calls].count("create_refund") == 1
    assert traces.get(third.trace_id)[-1]["status"] == "completed"

    state = run(store.load("user-1", "conv-1"))
    assert state.pending_action is None
    assert "refund_request" in state.completed_goals


def test_refund_rejection_does_not_execute_write_tool():
    runtime, _, calls, _ = build_runtime()

    run(runtime.run("user-1", "conv-reject", "退款 ORD-1001"))
    result = run(runtime.run("user-1", "conv-reject", "取消"))

    assert result.status == "completed"
    assert "取消" in result.response
    assert all(name != "create_refund" for name, _ in calls)


def test_ineligible_order_finishes_without_pending_action():
    runtime, store, calls, _ = build_runtime(eligible=False)

    result = run(runtime.run("user-1", "conv-expired", "退款 ORD-1001"))

    assert result.status == "completed"
    assert "无法发起退款" in result.response
    assert [name for name, _ in calls] == [
        "query_order",
        "check_refund_eligibility",
    ]
    state = run(store.load("user-1", "conv-expired"))
    assert state.pending_action is None


def test_missing_order_finishes_without_creating_refund():
    runtime, _, calls, _ = build_runtime()

    result = run(runtime.run(
        "user-1",
        "conv-not-found",
        "退款 ORD-9999",
    ))

    assert result.status == "completed"
    assert "没有找到" in result.response
    assert [name for name, _ in calls] == ["query_order"]


def test_tool_failure_returns_failed_handoff_status():
    runtime, _, _, traces = build_runtime(fail_query=True)

    result = run(runtime.run("user-1", "conv-fail", "退款 ORD-1001"))

    assert result.status == "failed"
    assert result.escalated is True
    assert traces.get(result.trace_id)[-1]["status"] == "failed"


def test_chat_response_keeps_legacy_fields_and_accepts_new_fields():
    response = ChatResponse(
        conv_id="conv-1",
        response="ok",
        intent="refund_request",
        agent_type="after_sales",
        escalated=False,
        latency_ms=1.2,
    )
    payload = response.model_dump()

    assert {
        "conv_id",
        "response",
        "intent",
        "agent_type",
        "escalated",
        "latency_ms",
        "knowledge_used",
    }.issubset(payload)
    assert payload["status"] == "completed"
    assert payload["missing_slots"] == []
    assert payload["citations"] == []


def test_chat_endpoint_delegates_to_state_runtime_and_keeps_conv_id(monkeypatch):
    class MemoryContext:
        recent_messages = []

    class FakeMemory:
        def __init__(self):
            self.messages = []
            self.events = []

        async def get_context(self, user_id, conv_id, query):
            return MemoryContext()

        async def add_message(self, user_id, conv_id, role, content):
            self.messages.append((user_id, conv_id, role.value, content))

        async def record_episodic_event(
            self, user_id, conv_id, event_type, summary=None, metadata=None,
        ):
            self.events.append((user_id, conv_id, event_type.value, metadata))

    class FakeRuntime:
        async def run(self, user_id, conv_id, message, history=None):
            return CustomerTurnResult(
                trace_id="trace-1",
                response="请提供需要处理的订单号。",
                intent="refund_request",
                agent_type="after_sales",
                status="awaiting_user",
                escalated=False,
                latency_ms=2.5,
                missing_slots=["order_id"],
            )

    memory = FakeMemory()
    monkeypatch.setattr(api_main, "_memory", memory)
    monkeypatch.setattr(api_main, "_customer_runtime", FakeRuntime())

    response = run(api_main.chat(ChatRequest(
        message="我要退款",
        user_id="user-1",
        conv_id="conv-api",
    )))

    assert response.conv_id == "conv-api"
    assert response.trace_id == "trace-1"
    assert response.status == "awaiting_user"
    assert response.missing_slots == ["order_id"]
    assert [item[3] for item in memory.messages] == [
        "我要退款",
        "请提供需要处理的订单号。",
    ]


def test_policy_question_uses_rag_and_returns_citation():
    runtime, _, calls, _ = build_runtime()

    result = run(runtime.run("user-1", "conv-policy", "退款政策是什么"))

    assert result.status == "completed"
    assert result.knowledge_used is True
    assert result.citations[0]["source"] == "policy.md"
    assert [name for name, _ in calls] == ["rag_search"]


def test_trace_store_returns_typed_public_events():
    runtime, _, _, traces = build_runtime()
    result = run(runtime.run("user-1", "conv-trace", "我要退款"))

    events = traces.get(result.trace_id)

    assert [event["event"] for event in events] == [
        "understanding",
        "turn_end",
    ]
    assert all("timestamp" in event for event in events)
    assert events[-1]["latency_ms"] >= 0


def test_trace_records_agent_tool_state_latency_and_trimmed_result_summary():
    runtime, _, _, traces = build_runtime()

    result = run(runtime.run("user-1", "conv-trace-tool", "退款 ORD-1001"))
    events = traces.get(result.trace_id)
    agent_event = next(event for event in events if event["event"] == "agent_result")
    turn_end = events[-1]

    assert agent_event["agent"] == "after_sales"
    assert agent_event["tools"] == [
        "query_order",
        "check_refund_eligibility",
    ]
    assert [item["name"] for item in agent_event["result_summary"]] == [
        "query_order",
        "check_refund_eligibility",
    ]
    assert turn_end["execution_state"] == "awaiting_confirmation"
    assert turn_end["state_path"]
    assert turn_end["latency_ms"] >= 0

    rendered = str(events).lower()
    for forbidden in (
        "退款 ord-1001",
        "full_prompt",
        "chain_of_thought",
        "hidden_reasoning",
    ):
        assert forbidden not in rendered


def test_trace_records_rag_result_summary_without_query_or_content():
    runtime, _, _, traces = build_runtime()

    result = run(runtime.run("user-1", "conv-trace-rag", "退款政策是什么"))
    events = traces.get(result.trace_id)
    rag_event = next(event for event in events if event["event"] == "rag_retrieval")

    assert rag_event["result_summary"][0]["name"] == "rag_search"
    preview = rag_event["result_summary"][0]["data_preview"]
    assert "policy.md" in preview
    assert "退款政策是什么" not in preview
    assert "购买后七天内" not in preview


def test_trace_endpoint_returns_events_and_rejects_unknown_trace(monkeypatch):
    _, _, _, traces = build_runtime()
    traces.append("trace-known", {"event": "turn_end", "latency_ms": 1.0})
    monkeypatch.setattr(api_main, "_trace_store", traces)

    response = run(api_main.get_trace("trace-known"))

    assert response["trace_id"] == "trace-known"
    assert response["events"][0]["event"] == "turn_end"
