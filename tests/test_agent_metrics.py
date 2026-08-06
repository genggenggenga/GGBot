import json
import asyncio
from pathlib import Path

import pytest

from evaluation.agent_metrics import (
    citation_precision,
    compare_retrievers,
    faithfulness_rate,
    joint_goal_accuracy,
    mean_reciprocal_rank,
    recall_at_k,
    slot_f1,
    task_completion_rate,
    tool_call_accuracy,
)
from monitor.performance_monitor import PerformanceMonitor


def test_slot_and_joint_goal_metrics():
    expected = [{"order_id": "ORD-1"}, {"order_id": "ORD-2"}]
    predicted = [{"order_id": "ORD-1"}, {"order_id": "WRONG"}]

    scores = slot_f1(expected, predicted)

    assert scores["precision"] == pytest.approx(0.5)
    assert scores["recall"] == pytest.approx(0.5)
    assert scores["f1"] == pytest.approx(0.5)
    assert joint_goal_accuracy(expected, predicted) == pytest.approx(0.5)


def test_retrieval_metrics_and_ablation_report():
    relevant = [{"a"}, {"b"}]
    ranked = [["a", "x"], ["x", "b"]]

    assert recall_at_k(relevant, ranked, 1) == pytest.approx(0.5)
    assert mean_reciprocal_rank(relevant, ranked) == pytest.approx(0.75)

    report = compare_retrievers([
        {"mode": "dense", "relevant_ids": ["a"], "ranked_ids": ["x", "a"]},
        {"mode": "hybrid", "relevant_ids": ["a"], "ranked_ids": ["a", "x"]},
        {"mode": "rerank", "relevant_ids": ["a"], "ranked_ids": ["a", "x"]},
    ])
    assert report["hybrid"]["mrr"] > report["dense"]["mrr"]
    assert report["rerank"]["recall_at_5"] == 1.0


def test_tool_and_task_metrics():
    tool_scores = tool_call_accuracy([
        {
            "expected_tool": "query_order",
            "predicted_tool": "query_order",
            "expected_params": {"order_id": "1"},
            "predicted_params": {"order_id": "1"},
        },
        {
            "expected_tool": "track_package",
            "predicted_tool": "query_order",
            "expected_params": {"order_id": "2"},
            "predicted_params": {"order_id": "2"},
        },
    ])

    assert tool_scores == {
        "selection_accuracy": 0.5,
        "parameter_accuracy": 1.0,
    }
    assert task_completion_rate([
        {"completed": True},
        {"completed": False},
    ]) == 0.5
    grounded = [
        {"grounded": True, "citations": [{"supported": True}]},
        {"grounded": False, "citations": [{"supported": False}]},
    ]
    assert citation_precision(grounded) == 0.5
    assert faithfulness_rate(grounded) == 0.5


def test_fixed_evaluation_dataset_has_required_coverage():
    path = Path("data/eval/customer_agent_cases.json")
    cases = json.loads(path.read_text(encoding="utf-8"))
    categories = {case["category"] for case in cases}

    assert 50 <= len(cases) <= 100
    assert {
        "slot_fill",
        "correction",
        "switch",
        "tool_failure",
        "no_answer",
        "compound",
    }.issubset(categories)


def test_monitor_reads_main_runtime_and_tool_registry_stats():
    class Runtime:
        def get_stats(self):
            return {
                "after_sales": {
                    "total": 3,
                    "success_rate": 1.0,
                    "avg_ms": 12.0,
                    "routing_score": 1.0,
                },
            }

    class Registry:
        def get_stats(self):
            return {
                "create_refund": {
                    "total": 1,
                    "success_rate": 1.0,
                    "avg_latency_ms": 5.0,
                    "consecutive_fails": 0,
                    "circuit_state": "closed",
                },
            }

    monitor = PerformanceMonitor(
        runtime=Runtime(),
        tool_registry=Registry(),
        interval_s=60,
    )
    asyncio.run(monitor._collect())
    summary = monitor.summary()

    assert summary["agent_stats"]["after_sales"]["total"] == 3
    assert summary["tool_stats"]["create_refund"]["total"] == 1


def test_monitor_deduplicates_active_threshold_alerts():
    class Runtime:
        def get_stats(self):
            return {
                "after_sales": {
                    "total": 12,
                    "success_rate": 0.5,
                    "avg_ms": 10.0,
                    "routing_score": 0.5,
                },
            }

    class Registry:
        def get_stats(self):
            return {}

    monitor = PerformanceMonitor(
        runtime=Runtime(),
        tool_registry=Registry(),
    )
    asyncio.run(monitor._collect())
    asyncio.run(monitor._collect())

    alerts = monitor.summary()["active_alerts"]
    assert len(alerts) == 1
    assert alerts[0]["metric"] == "agent_success_rate:after_sales"


# ── SubTask 12.1: Trace observation summaries and scrubbing ─────────────────

from core.trace_store import (
    TraceEvent,
    TraceStore,
    _FORBIDDEN_KEYS,
    _scrub_event,
    summarize_observations,
)
from core.agent_models import Observation


def test_summarize_observations_trims_data():
    long_data = {"title": "x" * 500}
    obs = Observation(
        source="tool", name="query_order", success=True, data=long_data,
    )
    summaries = summarize_observations([obs])
    assert len(summaries) == 1
    s = summaries[0]
    assert s["source"] == "tool"
    assert s["name"] == "query_order"
    assert s["success"] is True
    # data_preview must be trimmed
    assert len(s["data_preview"]) <= 130  # 120 + "..."
    assert s["data_preview"].endswith("...")


def test_summarize_observations_error_preview():
    obs = Observation(
        source="tool", name="bad_tool", success=False,
        data=None, error="timeout after 30s: " + "x" * 200,
    )
    summaries = summarize_observations([obs])
    s = summaries[0]
    assert s["success"] is False
    assert s["error_preview"] == "operation_failed"


def test_summarize_observations_no_data():
    obs = Observation(
        source="tool", name="rag_search", success=True, data=None,
    )
    summaries = summarize_observations([obs])
    s = summaries[0]
    assert "data_preview" not in s
    assert "error_preview" not in s


def test_summarize_observations_dict_input():
    """summarize_observations should also accept plain dicts."""
    obs_dict = {
        "source": "rag",
        "name": "rag_search",
        "success": True,
        "data": {"hits": 3},
    }
    summaries = summarize_observations([obs_dict])
    assert summaries[0]["name"] == "rag_search"
    assert summaries[0]["data_preview"] == '{"hits": 3}'


def test_summarize_observations_recursively_removes_sensitive_fields():
    secret = "用户原文和隐藏推理不应出现"
    obs = Observation(
        source="rag",
        name="rag_search",
        success=True,
        data={
            "query": secret,
            "hits": 2,
            "metadata": {
                "prompt": secret,
                "chain_of_thought": secret,
                "source": "policy.md",
                "content": secret,
            },
        },
    )

    summary = summarize_observations([obs])[0]["data_preview"]

    assert secret not in summary
    assert "query" not in summary
    assert "prompt" not in summary
    assert "chain_of_thought" not in summary
    assert "policy.md" in summary


def test_summarize_observations_limits_result_count():
    observations = [
        Observation(
            source="tool",
            name=f"tool_{index}",
            success=True,
            data={"status": "ok"},
        )
        for index in range(20)
    ]

    summaries = summarize_observations(observations)

    assert len(summaries) == 8


def test_scrub_event_removes_forbidden_keys():
    event = {
        "event": "understanding",
        "user_message": "我要退款",
        "prompt": "System: 你是客服...",
        "full_prompt": "...",
        "hidden_reasoning": "think step by step...",
        "reasoning_chain": "chain...",
        "raw_llm_output": "{...}",
        "intent": "refund_request",
    }
    scrubbed = _scrub_event(event)
    assert "intent" in scrubbed
    for key in _FORBIDDEN_KEYS:
        assert key not in scrubbed, f"forbidden key {key} leaked into trace"


def test_scrub_event_removes_nested_forbidden_keys():
    secret = "sensitive"
    scrubbed = _scrub_event({
        "event": "agent_result",
        "result_summary": [{
            "data": {
                "query": secret,
                "reasoning": secret,
                "status": "paid",
            },
        }],
    })
    rendered = json.dumps(scrubbed)
    assert secret not in rendered
    assert scrubbed["result_summary"][0]["data"]["status"] == "paid"


def test_trace_store_scrubs_on_append():
    store = TraceStore()
    store.append("t1", {
        "event": "understanding",
        "message": "用户原文不应出现",
        "prompt": "完整 prompt 不应出现",
        "hidden_reasoning": "隐藏推理不应出现",
        "intent": "refund_request",
    })
    events = store.get("t1")
    assert len(events) == 1
    for key in _FORBIDDEN_KEYS:
        assert key not in events[0], f"forbidden key {key} in stored trace"


def test_trace_store_uses_top_level_allowlist_for_business_fields():
    store = TraceStore()
    store.append("t1", {
        "event": "agent_result",
        "agent": "order",
        "success": True,
        "order_id": "ORD-SECRET",
        "user_id": "user-secret",
        "amount": 299.0,
        "tools": ["query_order"],
    })

    event = store.get("t1")[0]

    assert event["tools"] == ["query_order"]
    assert "order_id" not in event
    assert "user_id" not in event
    assert "amount" not in event


def test_observation_preview_uses_safe_metadata_allowlist():
    observation = Observation(
        source="tool",
        name="query_order",
        success=True,
        data={
            "order_id": "ORD-SECRET",
            "user_id": "user-secret",
            "amount": 299.0,
            "status": "paid",
            "source": "orders",
        },
    )

    preview = summarize_observations([observation])[0]["data_preview"]

    assert "paid" in preview
    assert "orders" in preview
    assert "ORD-SECRET" not in preview
    assert "user-secret" not in preview
    assert "299" not in preview


def test_trace_event_allows_observation_summaries():
    ev = TraceEvent.model_validate({
        "event": "agent_result",
        "agent": "after_sales",
        "success": True,
        "observation_summaries": [
            {"source": "tool", "name": "query_order", "success": True,
             "data_preview": "{'order_id': 'ORD-1001'}"},
        ],
    })
    assert ev.observation_summaries is not None
    assert ev.observation_summaries[0]["name"] == "query_order"


def test_trace_event_allows_result_summary():
    event = TraceEvent.model_validate({
        "event": "agent_result",
        "agent": "after_sales",
        "success": True,
        "result_summary": [{
            "source": "tool",
            "name": "query_order",
            "success": True,
            "data_preview": '{"status": "paid"}',
        }],
    })
    assert event.result_summary is not None
    assert event.result_summary[0]["name"] == "query_order"
