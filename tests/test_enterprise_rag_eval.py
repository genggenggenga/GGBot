import json

from evaluation.enterprise_rag_eval import (
    run_enterprise_rag_eval,
    write_reports,
)


def test_enterprise_rag_eval_covers_markdown_versions_and_metadata():
    report = run_enterprise_rag_eval()

    assert report.chunk_count == 52
    assert report.version_counts == {"v1": 25, "v2": 27}
    assert report.chunk_type_counts["guardrail"] == 6
    assert report.duplicate_chunk_ids == []
    assert report.metadata_errors == []
    assert report.parent_link_errors == []
    assert report.summary["recall_at_5"] >= 0.9
    assert report.summary["guardrail_recall"] == 1.0
    assert report.summary["parent_context_recall"] == 1.0
    assert report.summary["version_accuracy"] == 1.0
    assert report.summary["temporal_comparison_accuracy"] == 1.0
    assert report.summary["rerank_fallback_success_rate"] == 1.0


def test_enterprise_rag_eval_writes_reports(tmp_path):
    report = run_enterprise_rag_eval()
    json_path = tmp_path / "enterprise_rag.json"
    markdown_path = tmp_path / "enterprise_rag.md"

    write_reports(report, json_path=json_path, markdown_path=markdown_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["chunk_count"] == 52
    assert "Enterprise RAG Evaluation" in markdown
    assert "guardrail_recall" in markdown
