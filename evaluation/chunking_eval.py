"""Golden evaluation for the real GGBot parsing and chunking pipeline."""
from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from rag.indexes import BM25Index
from rag.loaders import ChunkingConfig, chunk_sections, load_document
from rag.models import DocumentChunk
from rag.tokenization import count_tokens


_ROOT = pathlib.Path(__file__).parent.parent
_DEFAULT_CASES = _ROOT / "data" / "eval" / "rag_chunking_cases.json"
_REPORTS_DIR = _ROOT / "data" / "eval" / "reports" / datetime.now().date().isoformat()
_DEFAULT_JSON_REPORT = _REPORTS_DIR / "chunking.json"
_DEFAULT_MD_REPORT = _REPORTS_DIR / "chunking.md"


@dataclass
class ChunkingCaseResult:
    case_id: str
    passed: bool
    rank: Optional[int]
    relevant_doc_id: str
    matched_chunk_id: Optional[str]
    expected_section: str
    actual_section: Optional[str]
    evidence_preserved: bool


@dataclass
class ChunkingEvalReport:
    generated_at: str
    dataset: str
    reproduce_command: str
    strategy: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    document_count: int
    chunk_count: int
    case_count: int
    retrieval_hit_rate: float
    mrr: float
    evidence_preservation_rate: float
    section_accuracy: float
    max_chunk_tokens: int
    budget_violations: int
    cases: List[ChunkingCaseResult]


def run_chunking_eval(
    *,
    dataset_path: pathlib.Path = _DEFAULT_CASES,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    top_k: int = 5,
) -> ChunkingEvalReport:
    """Run real loaders, token-budget chunking, and BM25 over golden cases."""
    env_config = ChunkingConfig.from_env()
    config = ChunkingConfig(
        chunk_size=(
            env_config.chunk_size if chunk_size is None else chunk_size
        ),
        chunk_overlap=(
            env_config.chunk_overlap
            if chunk_overlap is None
            else chunk_overlap
        ),
    )
    if top_k < 1:
        raise ValueError("top_k must be positive")
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = list(payload["cases"])
    if payload.get("generated_case_set") == "chunking_v1":
        from evaluation.chunking_expansion import build_chunking_v1_expansion

        cases.extend(build_chunking_v1_expansion(cases))
    elif payload.get("generated_case_set"):
        raise ValueError(
            f"unsupported chunking generated case set: "
            f"{payload['generated_case_set']!r}"
        )
    chunks = _load_chunks(payload["documents"], config)
    index = BM25Index()
    index.add(chunks)

    results = [
        _evaluate_case(case, chunks, index, top_k)
        for case in cases
    ]
    ranks = [result.rank for result in results if result.rank is not None]
    section_cases = [
        result for result in results if result.expected_section
    ]
    token_counts = [count_tokens(chunk.content) for chunk in chunks]
    command = (
        ".venv/bin/python -m evaluation.chunking_eval "
        f"--chunk-size {config.chunk_size} "
        f"--chunk-overlap {config.chunk_overlap}"
    )
    try:
        dataset = str(dataset_path.relative_to(_ROOT))
    except ValueError:
        dataset = str(dataset_path)
    return ChunkingEvalReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        dataset=dataset,
        reproduce_command=command,
        strategy=(
            "generic_recursive"
            "（load_document + chunk_sections，按 token 预算切分；"
            "GGKB 类型化整块策略见 enterprise_rag 评测）"
        ),
        chunk_size_tokens=config.chunk_size,
        chunk_overlap_tokens=config.chunk_overlap,
        document_count=len(payload["documents"]),
        chunk_count=len(chunks),
        case_count=len(results),
        retrieval_hit_rate=_ratio(
            sum(result.rank is not None for result in results),
            len(results),
        ),
        mrr=round(
            sum(1.0 / rank for rank in ranks) / len(results),
            4,
        ) if results else 0.0,
        evidence_preservation_rate=_ratio(
            sum(result.evidence_preserved for result in results),
            len(results),
        ),
        section_accuracy=_ratio(
            sum(
                result.actual_section == result.expected_section
                for result in section_cases
            ),
            len(section_cases),
        ),
        max_chunk_tokens=max(token_counts, default=0),
        budget_violations=sum(
            token_count > config.chunk_size
            for token_count in token_counts
        ),
        cases=results,
    )


def _load_chunks(
    documents: List[Dict[str, Any]],
    config: ChunkingConfig,
) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        for document in documents:
            path = root / document["filename"]
            path.write_text(document["content"], encoding="utf-8")
            sections = [
                section.model_copy(update={
                    "source": document["filename"],
                    "metadata": {
                        **section.metadata,
                        "doc_id": document["doc_id"],
                    },
                })
                for section in load_document(path)
            ]
            chunks.extend(chunk_sections(
                sections,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            ))
    return chunks


def _evaluate_case(
    case: Dict[str, Any],
    chunks: List[DocumentChunk],
    index: BM25Index,
    top_k: int,
) -> ChunkingCaseResult:
    expected_text = _normalize(case["expected_text"])
    relevant_doc_id = case["relevant_doc_id"]
    evidence_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata.get("doc_id") == relevant_doc_id
        and expected_text in _normalize(chunk.content)
    ]
    hits = index.search(case["query"], top_k)
    matched_rank = None
    matched_chunk = None
    for rank, hit in enumerate(hits, start=1):
        if (
            hit.chunk.metadata.get("doc_id") == relevant_doc_id
            and expected_text in _normalize(hit.chunk.content)
        ):
            matched_rank = rank
            matched_chunk = hit.chunk
            break

    expected_section = case.get("expected_section", "")
    section_matches = (
        not expected_section
        or (
            matched_chunk is not None
            and matched_chunk.section == expected_section
        )
    )
    return ChunkingCaseResult(
        case_id=case["id"],
        passed=matched_rank is not None and section_matches,
        rank=matched_rank,
        relevant_doc_id=relevant_doc_id,
        matched_chunk_id=(
            matched_chunk.chunk_id if matched_chunk is not None else None
        ),
        expected_section=expected_section,
        actual_section=(
            matched_chunk.section if matched_chunk is not None else None
        ),
        evidence_preserved=bool(evidence_chunks),
    )


def write_reports(
    report: ChunkingEvalReport,
    *,
    json_path: pathlib.Path = _DEFAULT_JSON_REPORT,
    markdown_path: pathlib.Path = _DEFAULT_MD_REPORT,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows = "\n".join(
        f"| {case.case_id} | {'PASS' if case.passed else 'FAIL'} | "
        f"{case.rank or '-'} | {case.actual_section or '-'} |"
        for case in report.cases
    )
    markdown_path.write_text(
        "\n".join([
            "# GGBot Chunking 评测报告",
            "",
            f"- 生成时间：{report.generated_at}",
            f"- 数据集：`{report.dataset}`",
            f"- 切块策略：{report.strategy}",
            f"- 参数：chunk={report.chunk_size_tokens} tokens，"
            f"overlap={report.chunk_overlap_tokens} tokens",
            f"- 文档/Chunk/用例：{report.document_count} / "
            f"{report.chunk_count} / {report.case_count}",
            f"- Retrieval Hit Rate@5：{report.retrieval_hit_rate:.4f}",
            f"- MRR：{report.mrr:.4f}",
            f"- Evidence Preservation："
            f"{report.evidence_preservation_rate:.4f}",
            f"- Section Accuracy：{report.section_accuracy:.4f}",
            f"- 最大 Chunk：{report.max_chunk_tokens} tokens",
            f"- Token 预算违规：{report.budget_violations}",
            "",
            "## 用例",
            "",
            "| Case | 结果 | Rank | Section |",
            "|---|---:|---:|---|",
            rows,
            "",
            "## 复现",
            "",
            "```bash",
            report.reproduce_command,
            "```",
            "",
        ]),
        encoding="utf-8",
    )


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, default=_DEFAULT_CASES)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    report = run_chunking_eval(
        dataset_path=args.dataset,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
    )
    if not args.no_write:
        write_reports(report)
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
