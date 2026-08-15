import json
from types import SimpleNamespace

import pytest

from evaluation.agent_metrics import (
    abstention_metrics,
    citation_recall,
    forbidden_tool_rate,
    postcondition_success_rate,
    tool_trace_exact_match,
)
from evaluation.datasets import load_dataset
from evaluation.judge import JudgeInput, LLMJudge
from evaluation.local_eval_runner import run_local_eval


def test_versioned_suites_load_with_metadata():
    smoke = load_dataset("smoke")
    golden = load_dataset("golden")
    bad_cases = load_dataset("bad_cases")

    assert len(smoke.cases) == 90
    assert golden.version == "golden-v1-candidate"
    assert golden.corpus_version == "corpus-v1"
    assert len(golden.cases) == 280
    assert bad_cases.cases[0].risk_level == "critical"
    assert len(bad_cases.cases) == 50


def test_workflow_and_grounding_metrics():
    workflows = [{
        "expected_tool_trace": ["query_order", "create_refund"],
        "observed_tools": ["query_order", "create_refund"],
        "forbidden_tools": ["create_ticket"],
        "expected_postconditions": {"refund_created": True},
        "observed_postconditions": {"refund_created": True},
    }]
    assert tool_trace_exact_match(workflows) == 1.0
    assert forbidden_tool_rate(workflows) == 0.0
    assert postcondition_success_rate(workflows) == 1.0
    assert citation_recall([{
        "required_evidence_ids": ["refund-window"],
        "citations": [{"chunk_id": "refund-window"}],
    }]) == 1.0
    assert abstention_metrics([{
        "must_abstain": True,
        "abstained": True,
    }]) == {"precision": 1.0, "recall": 1.0}


@pytest.mark.asyncio
async def test_golden_and_bad_case_suites_execute():
    golden = await run_local_eval(suite="golden")
    bad_cases = await run_local_eval(suite="bad_cases")

    assert golden.suite == "golden"
    assert golden.dataset["corpus_version"] == "corpus-v1"
    assert "tool_trace_exact_match" in golden.summary
    assert bad_cases.suite == "bad_cases"
    assert bad_cases.summary["confirmation_safety_rate"] == 1.0


@pytest.mark.asyncio
async def test_llm_judge_uses_structured_scores_without_network():
    payload = json.dumps({
        "relevance": 1,
        "accuracy": 0.9,
        "completeness": 0.8,
        "helpfulness": 0.9,
        "safety": 1,
        "groundedness": 0.8,
        "violations": [],
    })

    class Messages:
        async def create(self, **kwargs):
            assert kwargs["temperature"] == 0.0
            return SimpleNamespace(content=[{"type": "text", "text": payload}])

    judge = LLMJudge(SimpleNamespace(messages=Messages()), "test-model")
    scores = await judge.judge(JudgeInput(
        user_question="退款了吗",
        candidate_response="退款申请已提交。",
        tool_observations=[{"tool": "create_refund", "success": True}],
    ))

    assert scores.judge_failed is False
    assert scores.safety == 1.0
    assert scores.model_dump()["overall"] > 0.8
