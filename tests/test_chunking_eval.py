import json

from evaluation.chunking_eval import run_chunking_eval, write_reports


def test_chunking_eval_runs_real_pipeline():
    report = run_chunking_eval(chunk_size=128, chunk_overlap=16)

    assert report.document_count == 6
    assert report.case_count == 12
    assert report.chunk_count >= report.document_count
    assert report.budget_violations == 0
    assert report.max_chunk_tokens <= 128
    assert report.evidence_preservation_rate == 1.0
    assert report.retrieval_hit_rate >= 0.8


def test_chunking_eval_writes_reproducible_reports(tmp_path):
    report = run_chunking_eval(chunk_size=96, chunk_overlap=12)
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    write_reports(
        report,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["chunk_size_tokens"] == 96
    assert payload["chunk_overlap_tokens"] == 12
    assert "GGBot Chunking 评测报告" in markdown
    assert report.reproduce_command in markdown
