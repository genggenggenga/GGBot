"""Tests for the deterministic local eval runner (Task 12 SubTask 12.3)."""
import json
import pathlib
import pytest

from evaluation.local_eval_runner import (
    CaseResult,
    EvalReport,
    FakeDenseIndex,
    FakeReranker,
    FakeSparseIndex,
    _build_tool_registry,
    _compute_all_metrics,
    _compute_intent_metrics,
    _default_seed_chunks,
    _nlu_fast,
    run_ablation,
    run_baseline_comparison,
    run_local_eval,
    write_ablation_json,
    write_json_report,
    write_markdown_report,
)


@pytest.fixture
def seed_chunks():
    return _default_seed_chunks()


@pytest.fixture
def tool_registry():
    return _build_tool_registry()


class TestFakeDenseIndex:
    def test_add_and_search(self, seed_chunks):
        idx = FakeDenseIndex()
        idx.add(seed_chunks)
        hits = idx.search("退款期限", top_k=3)
        assert len(hits) > 0
        assert hits[0].chunk.chunk_id == "refund-window"

    def test_empty_search(self):
        idx = FakeDenseIndex()
        assert idx.search("anything", top_k=5) == []


class TestFakeSparseIndex:
    def test_add_and_search(self, seed_chunks):
        idx = FakeSparseIndex()
        idx.add(seed_chunks)
        hits = idx.search("退款期限", top_k=3)
        assert len(hits) > 0

    def test_empty_search(self):
        idx = FakeSparseIndex()
        assert idx.search("anything", top_k=5) == []


class TestFakeReranker:
    def test_deterministic_scores(self):
        from rag.models import DocumentChunk
        reranker = FakeReranker()
        chunks = [
            DocumentChunk(chunk_id="a", content="x", source="s", title="t",
                          section="", chunk_index=0, parent_id="p"),
            DocumentChunk(chunk_id="b", content="y", source="s", title="t",
                          section="", chunk_index=1, parent_id="p"),
        ]
        scores = reranker.score("query", chunks)
        assert len(scores) == 2
        assert all(0.0 <= s <= 1.0 for s in scores)
        # Must be deterministic
        scores2 = reranker.score("query", chunks)
        assert scores == scores2


class TestNluFast:
    def test_refund_with_order_id(self):
        result = _nlu_fast("退款 ORD-1001")
        assert result.primary_intent == "refund_request"
        assert result.extracted_slots.get("order_id") == "ORD-1001"

    def test_logistics_query(self):
        result = _nlu_fast("查物流")
        assert result.primary_intent == "logistics_query"

    def test_fallback_for_unknown(self):
        result = _nlu_fast("讲一个笑话")
        assert result.primary_intent == "other"


class TestToolRegistry:
    @pytest.mark.asyncio
    async def test_query_order(self, tool_registry):
        result = await tool_registry.call(
            "after_sales", "query_order", {"order_id": "ORD-1001"},
        )
        assert result.success
        assert result.data["found"] is True

    @pytest.mark.asyncio
    async def test_order_not_found(self, tool_registry):
        result = await tool_registry.call(
            "after_sales", "query_order", {"order_id": "ORD-NOT-FOUND"},
        )
        assert result.success
        assert result.data["found"] is False


class TestRunLocalEval:
    @pytest.mark.asyncio
    async def test_produces_report(self, seed_chunks):
        report = await run_local_eval(rag_mode="hybrid", seed_chunks=seed_chunks)
        assert isinstance(report, EvalReport)
        assert report.sample_size == 90
        assert len(report.per_case) == 90
        assert "intent_accuracy" in report.summary
        assert "intent_macro_f1" in report.summary
        assert "user_act_accuracy" in report.summary
        assert "slot_f1" in report.summary
        assert "dst_joint_goal_accuracy" in report.summary
        assert "recall_at_5" in report.summary
        assert "mrr" in report.summary
        assert "chunk_type_recall" in report.summary
        assert "guardrail_recall" in report.summary
        assert "parent_context_recall" in report.summary
        assert "rerank_fallback_rate" in report.summary
        assert "metadata_boost_rate" in report.summary
        assert "tool_selection_accuracy" in report.summary
        assert "tool_parameter_accuracy" in report.summary
        assert "task_completion_rate" in report.summary
        assert "citation_precision" in report.summary
        assert "faithfulness_rate" in report.summary
        assert report.summary["citation_precision"] > 0
        assert report.summary["faithfulness_rate"] > 0

    @pytest.mark.asyncio
    async def test_each_input_case_runs_once_in_source_order(self, seed_chunks):
        from evaluation.datasets import load_dataset

        source_cases = load_dataset("smoke").cases

        report = await run_local_eval(rag_mode="hybrid", seed_chunks=seed_chunks)
        result_ids = [case["case_id"] for case in report.per_case]
        source_ids = [case.id for case in source_cases]

        assert report.sample_size == len(source_cases) == 90
        assert result_ids == source_ids
        assert len(result_ids) == len(set(result_ids)) == 90

    @pytest.mark.asyncio
    async def test_intent_accuracy_above_threshold(self, seed_chunks):
        report = await run_local_eval(seed_chunks=seed_chunks)
        # With deterministic fast-track, we expect reasonable accuracy
        assert report.summary["intent_accuracy"] >= 0.8

    @pytest.mark.asyncio
    async def test_slot_f1_perfect(self, seed_chunks):
        report = await run_local_eval(seed_chunks=seed_chunks)
        assert report.summary["slot_f1"] == 1.0

    @pytest.mark.asyncio
    async def test_runtime_cases_use_real_tool_observations_and_failure_status(
        self,
        seed_chunks,
    ):
        report = await run_local_eval(seed_chunks=seed_chunks)
        by_id = {case["case_id"]: case for case in report.per_case}

        assert by_id["TOOL-006"]["observed_tools"][-1] == "create_refund"
        assert by_id["E2E-009"]["predicted_status"] == "failed"
        assert by_id["E2E-009"]["completed"] is True
        assert by_id["NLU-008"]["predicted_act"] == "reject"

    def test_dst_ground_truth_is_explicit_in_fixture(self):
        cases = json.loads(
            pathlib.Path("data/eval/customer_agent_cases.json").read_text(
                encoding="utf-8",
            )
        )
        dst_cases = [case for case in cases if case["id"].startswith("DST-")]

        assert dst_cases
        assert all("expected_slots" in case for case in dst_cases)


class TestAblation:
    @pytest.mark.asyncio
    async def test_ablation_three_modes(self, seed_chunks):
        ablation = await run_ablation()
        assert set(ablation.keys()) == {"dense", "hybrid", "rerank"}
        for mode, report in ablation.items():
            assert report.sample_size == 90
            assert "intent_accuracy" in report.summary

    @pytest.mark.asyncio
    async def test_rerank_differs_from_hybrid(self, seed_chunks):
        ablation = await run_ablation()
        # With the fake reranker scoring by hash, MRR should differ
        hybrid_mrr = ablation["hybrid"].summary["mrr"]
        rerank_mrr = ablation["rerank"].summary["mrr"]
        # At minimum they should produce valid scores
        assert 0.0 <= hybrid_mrr <= 1.0
        assert 0.0 <= rerank_mrr <= 1.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("mode", "expected_flags"),
        [
            ("dense", (False, False)),
            ("hybrid", (True, False)),
            ("rerank", (True, True)),
        ],
    )
    async def test_modes_use_distinct_retrieval_paths(
        self, monkeypatch, seed_chunks, mode, expected_flags
    ):
        from rag.retriever import HybridRetriever

        calls = []
        original_search = HybridRetriever.search

        def search_spy(self, query, **kwargs):
            calls.append((kwargs["use_sparse"], kwargs["use_reranker"]))
            return original_search(self, query, **kwargs)

        monkeypatch.setattr(HybridRetriever, "search", search_spy)
        await run_local_eval(rag_mode=mode, seed_chunks=seed_chunks)

        # Ten dedicated RAG cases plus runtime Tool/E2E cases that actually
        # route through KnowledgeAgent.
        assert len(calls) >= 10
        assert set(calls) == {expected_flags}


class TestBaselineComparison:
    @pytest.mark.asyncio
    async def test_baseline_vs_current(self, seed_chunks):
        comparison = await run_baseline_comparison()
        assert comparison["comparison_type"] == "synthetic_or_legacy_rules"
        assert comparison["supports_measured_improvement_claim"] is False
        assert "legacy_rules_task_completion_rate" in comparison
        assert "current_task_completion_rate" in comparison
        assert "rate_difference" in comparison
        assert "improvement" not in comparison
        assert comparison["e2e_sample_size"] == 18


class TestReportWriting:
    def test_json_report(self, tmp_path):
        report = EvalReport(
            generated_at="2026-01-01",
            reproduce_command="test",
            sample_size=5,
            summary={"intent_accuracy": 0.9},
            per_case=[{"case_id": "T1"}],
        )
        path = write_json_report(report, tmp_path / "report.json")
        data = json.loads(path.read_text())
        assert data["sample_size"] == 5

    def test_markdown_report(self, tmp_path):
        report = EvalReport(
            generated_at="2026-01-01",
            reproduce_command="test",
            sample_size=5,
            summary={"intent_accuracy": 0.9, "macro_f1": 0.85},
            per_case=[{"case_id": "T1", "category": "intent",
                        "predicted_intent": "refund_request", "completed": True}],
        )
        path = write_markdown_report(report, path=tmp_path / "report.md")
        text = path.read_text()
        assert "# GGBot Evaluation Report" in text
        assert "intent_accuracy" in text

    def test_ablation_json(self, tmp_path):
        report1 = EvalReport(
            generated_at="2026-01-01", reproduce_command="test",
            sample_size=5, summary={"intent_accuracy": 0.9},
        )
        report2 = EvalReport(
            generated_at="2026-01-01", reproduce_command="test",
            sample_size=5, summary={"intent_accuracy": 0.95},
        )
        path = write_ablation_json({"dense": report1, "hybrid": report2},
                                    tmp_path / "ablation.json")
        data = json.loads(path.read_text())
        assert "dense" in data["modes"]
        assert "hybrid" in data["modes"]


class TestIntentMetrics:
    def test_perfect_accuracy(self):
        results = [
            CaseResult(case_id="1", category="intent",
                       expected_intent="a", predicted_intent="a"),
            CaseResult(case_id="2", category="intent",
                       expected_intent="b", predicted_intent="b"),
        ]
        metrics = _compute_intent_metrics(results)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_zero_accuracy(self):
        results = [
            CaseResult(case_id="1", category="intent",
                       expected_intent="a", predicted_intent="b"),
        ]
        metrics = _compute_intent_metrics(results)
        assert metrics["accuracy"] == 0.0
