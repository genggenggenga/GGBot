import asyncio
import inspect
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
    monkeypatch.setattr(api_main, "_orchestrator", orchestrator)
    monkeypatch.setattr(api_main, "_skill_manager", skills)

    assert run(api_main.health())["status"] == "ok"
    assert run(api_main.skills_summary()) == {"count": 3}
    assert run(api_main.reload_skills()) == {"count": 3}
    assert skills.reload_count == 1
    assert orchestrator.skill_manager is skills


def test_knowledge_add_and_stats_keep_legacy_tool_contract(monkeypatch):
    class KnowledgeBase:
        doc_count = 8

        def add_documents(self, documents):
            self.doc_count += len(documents)
            return len(documents)

        async def search_handler(self, params, context):
            return []

    kb = KnowledgeBase()
    tool = SimpleNamespace(handler=kb.search_handler)
    manager = SimpleNamespace(_tools={"knowledge_search": tool})
    monkeypatch.setattr(api_main, "_tool_manager", manager)

    result = run(api_main.add_knowledge(api_main.BatchDocInput(
        documents=[api_main.DocInput(title="退款", content="七天内可退")],
    )))
    stats = run(api_main.knowledge_stats())

    assert result["added_chunks"] == 1
    assert stats["total_chunks"] == 9


def test_existing_routes_are_still_registered():
    paths = {route.path for route in api_main.app.routes}

    assert {
        "/health",
        "/skills",
        "/skills/reload",
        "/chat",
        "/knowledge/add",
        "/knowledge/upload",
        "/knowledge/stats",
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


def test_eval_default_mode_keeps_legacy_evaluator(monkeypatch):
    class Evaluator:
        async def run(self, intent_cases, dialog_cases):
            return SimpleNamespace(
                pass_rate=1.0,
                total=1,
                passed=1,
                avg_scores={"intent_accuracy": 1.0},
                regressions=[],
                recommendations=[],
                results=[],
            )

    monkeypatch.setattr(api_main, "_evaluator", Evaluator())
    result = run(api_main.run_eval())

    assert result["pass_rate"] == 1.0
    assert result["total"] == 1
    assert "mode" not in result


def test_eval_rejects_unknown_mode():
    with pytest.raises(HTTPException) as exc_info:
        run(api_main.run_eval(api_main.EvalRunInput(mode="unknown")))

    assert exc_info.value.status_code == 400


def test_legacy_baseline_is_not_loaded_by_default():
    source = inspect.getsource(api_main._runtime_components)

    assert '"/app/data/eval/runtime_baseline.json"' in source
    assert '"/app/data/eval/baseline.json"' not in source
    assert not Path("data/eval/baseline.json").exists()
