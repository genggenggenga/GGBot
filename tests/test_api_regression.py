import asyncio
import inspect
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api import main as api_main


def run(coro):
    return asyncio.run(coro)


def test_health_and_skill_endpoints_remain_compatible(monkeypatch):
    class Orchestrator:
        def __init__(self):
            self.skill_manager = None

        def get_stats(self):
            return {"general": {"total": 1}}

        def set_skill_manager(self, manager):
            self.skill_manager = manager

    class Skills:
        def __init__(self):
            self.reload_count = 0

        def reload(self):
            self.reload_count += 1

        def summary(self):
            return {"count": 3}

    orchestrator = Orchestrator()
    skills = Skills()
    runtime = SimpleNamespace(
        get_stats=lambda: {"knowledge": {"total": 1}},
    )
    registry = SimpleNamespace(
        get_stats=lambda: {"rag_search": {"total": 1}},
        list_tools=lambda: [],
    )
    monkeypatch.setattr(api_main, "_orchestrator", orchestrator)
    monkeypatch.setattr(api_main, "_skill_manager", skills)
    monkeypatch.setattr(api_main, "_customer_runtime", runtime)
    monkeypatch.setattr(api_main, "_tool_registry", registry)
    monkeypatch.setattr(api_main, "_memory", object())
    monkeypatch.setattr(api_main, "_knowledge_runtime", object())
    monkeypatch.setattr(api_main, "_conversation_locks", object())

    health = run(api_main.health())
    assert health["status"] == "ok"
    assert health["agents"]["knowledge"]["total"] == 1
    assert health["tools"]["rag_search"]["total"] == 1
    assert run(api_main.skills_summary()) == {"count": 3}
    assert run(api_main.reload_skills()) == {"count": 3}
    assert skills.reload_count == 1
    assert orchestrator.skill_manager is skills


def test_health_rejects_partial_primary_runtime(monkeypatch):
    monkeypatch.setattr(api_main, "_customer_runtime", object())
    monkeypatch.setattr(api_main, "_tool_registry", None)
    monkeypatch.setattr(api_main, "_memory", object())
    monkeypatch.setattr(api_main, "_knowledge_runtime", object())
    monkeypatch.setattr(api_main, "_conversation_locks", object())

    with pytest.raises(HTTPException) as exc_info:
        run(api_main.health())

    assert exc_info.value.status_code == 503
    assert "tool_registry" in str(exc_info.value.detail)


def test_compat_tools_endpoint_returns_internal_rpc_contracts(monkeypatch):
    registry = SimpleNamespace(
        list_tools=lambda: [
            SimpleNamespace(
                name="commerce.query_order",
                description="Query an order",
                input_schema={
                    "type": "object",
                    "required": ["order_id"],
                },
                output_schema={"type": "object"},
            ),
            SimpleNamespace(
                name="after_sales.create_refund",
                description="Create refund",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
            ),
        ],
    )
    monkeypatch.setattr(api_main, "_tool_registry", registry)

    result = run(api_main.list_mcp_tools())

    assert result.total == 2
    assert result.tools[0].name == "commerce.query_order"
    assert result.tools[0].input_schema["required"] == ["order_id"]
    assert result.tools[0].annotations is None
    assert result.tools[1].description == "Create refund"


def test_tools_endpoint_rejects_uninitialized_registry(monkeypatch):
    monkeypatch.setattr(api_main, "_tool_registry", None)

    with pytest.raises(HTTPException) as exc_info:
        run(api_main.list_mcp_tools())

    assert exc_info.value.status_code == 503


def test_knowledge_add_and_stats_use_knowledge_runtime(monkeypatch):
    class KnowledgeBase:
        doc_count = 8

        def add_documents(self, documents):
            self.doc_count += len(documents)
            return len(documents)

        async def search_handler(self, params, context):
            return []

    kb = KnowledgeBase()
    runtime = SimpleNamespace(
        knowledge_base=kb,
        add_documents=kb.add_documents,
    )
    monkeypatch.setattr(api_main, "_knowledge_runtime", runtime)

    result = run(api_main.add_knowledge(api_main.BatchDocInput(
        documents=[api_main.DocInput(title="退款", content="七天内可退")],
    )))
    stats = run(api_main.knowledge_stats())

    assert result["added_chunks"] == 1
    assert stats["total_chunks"] == 9


def test_knowledge_add_passes_version_metadata(monkeypatch):
    received = []

    class KnowledgeBase:
        doc_count = 1

    class Runtime:
        knowledge_base = KnowledgeBase()

        def add_documents(self, documents):
            received.extend(documents)
            return 1

    monkeypatch.setattr(api_main, "_knowledge_runtime", Runtime())
    effective_at = datetime(2026, 8, 8, tzinfo=timezone.utc)

    result = run(api_main.add_knowledge(api_main.BatchDocInput(
        documents=[api_main.DocInput(
            title="退款政策",
            content="七天内可退",
            knowledge_id="refund-policy",
            version="v2",
            effective_at=effective_at,
        )],
    )))

    assert result["added_chunks"] == 1
    assert received[0]["metadata"]["knowledge_id"] == "refund-policy"
    assert received[0]["metadata"]["version"] == "v2"
    assert received[0]["metadata"]["effective_at"] == effective_at


def test_knowledge_version_lifecycle_endpoints_refresh_runtime(monkeypatch):
    class KnowledgeBase:
        def list_versions(self, knowledge_id):
            return [{"knowledge_id": knowledge_id, "version": "v2"}]

        def publish_version(self, knowledge_id, version, **kwargs):
            return {
                "knowledge_id": knowledge_id,
                "version": version,
                "status": "published",
            }

        def revoke_version(self, knowledge_id, version):
            return {
                "knowledge_id": knowledge_id,
                "version": version,
                "status": "revoked",
            }

    class Runtime:
        knowledge_base = KnowledgeBase()
        refresh_count = 0

        def refresh(self):
            self.refresh_count += 1

    runtime = Runtime()
    monkeypatch.setattr(api_main, "_knowledge_runtime", runtime)

    versions = run(api_main.list_knowledge_versions("refund-policy"))
    published = run(api_main.publish_knowledge_version(
        "refund-policy",
        "v2",
    ))
    revoked = run(api_main.revoke_knowledge_version(
        "refund-policy",
        "v2",
    ))

    assert versions["versions"][0]["version"] == "v2"
    assert published["status"] == "published"
    assert revoked["status"] == "revoked"
    assert runtime.refresh_count == 2


def test_existing_routes_are_still_registered():
    paths = {route.path for route in api_main.app.routes}

    assert {
        "/health",
        "/skills",
        "/skills/reload",
        "/mcp/tools",
        "/chat",
        "/knowledge/add",
        "/knowledge/upload",
        "/knowledge/stats",
        "/knowledge/{knowledge_id}/versions",
        "/knowledge/{knowledge_id}/versions/{version}/publish",
        "/knowledge/{knowledge_id}/versions/{version}/revoke",
        "/eval/run",
        "/traces/{trace_id}",
    }.issubset(paths)


def test_customer_agent_eval_returns_layered_summary(monkeypatch):
    async def fake_run_local_eval():
        return SimpleNamespace(
            generated_at="2026-08-03T00:00:00+00:00",
            reproduce_command="python -m evaluation.local_eval_runner",
            sample_size=50,
            summary={
                "intent_accuracy": 0.9,
                "slot_f1": 1.0,
                "dst_joint_goal_accuracy": 1.0,
            },
        )

    monkeypatch.setattr(
        "evaluation.local_eval_runner.run_local_eval",
        fake_run_local_eval,
    )
    result = run(api_main.run_eval(api_main.EvalRunInput(
        mode="customer_agent",
    )))

    assert result["mode"] == "customer_agent"
    assert result["sample_size"] == 50
    assert result["summary"]["slot_f1"] == 1.0
    assert result["generated_at"] == "2026-08-03T00:00:00+00:00"


def test_eval_default_mode_uses_customer_agent_runner(monkeypatch):
    async def fake_run_local_eval():
        return SimpleNamespace(
            generated_at="2026-08-07T00:00:00+00:00",
            reproduce_command="python -m evaluation.local_eval_runner",
            sample_size=50,
            summary={"task_completion_rate": 1.0},
        )

    monkeypatch.setattr(
        "evaluation.local_eval_runner.run_local_eval",
        fake_run_local_eval,
    )
    result = run(api_main.run_eval())

    assert result["mode"] == "customer_agent"
    assert result["sample_size"] == 50
    assert result["summary"]["task_completion_rate"] == 1.0


def test_search_uses_hybrid_knowledge_runtime(monkeypatch):
    class Retrieval:
        def model_dump(self, mode):
            return {
                "answered": True,
                "hits": [{"score": 0.9}],
                "citations": [{"source": "policy.md"}],
                "reason": None,
            }

    class Retriever:
        def search(self, query, **kwargs):
            assert query == "退款"
            assert kwargs["top_k"] == 3
            return Retrieval()

    monkeypatch.setattr(
        api_main,
        "_knowledge_runtime",
        SimpleNamespace(retriever=Retriever()),
    )

    result = run(api_main.search("退款", top_k=3))

    assert result["query"] == "退款"
    assert result["answered"] is True
    assert result["citations"][0]["source"] == "policy.md"


def test_eval_rejects_unknown_mode():
    with pytest.raises(HTTPException) as exc_info:
        run(api_main.run_eval(api_main.EvalRunInput(mode="unknown")))

    assert exc_info.value.status_code == 400


def test_legacy_baseline_is_not_loaded_by_default():
    source = inspect.getsource(api_main._runtime_components)

    assert '"/app/data/eval/runtime_baseline.json"' in source
    assert '"/app/data/eval/baseline.json"' not in source
    assert not Path("data/eval/baseline.json").exists()


def test_cli_reuses_primary_chat_runtime():
    source = inspect.getsource(api_main._cli)

    assert "await chat(ChatRequest(" in source
    assert "AgentOrchestrator" not in source


def test_chat_records_episodic_memory_inside_conversation_transaction():
    source = inspect.getsource(api_main._chat_locked)

    assert "await _memory.record_episodic_event(" in source
    assert "_spawn_background_task" not in source
