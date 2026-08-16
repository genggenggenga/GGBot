"""Enterprise RAG evaluation for annotated Markdown knowledge assets."""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from rag.indexes import BM25Index
from rag.markdown_ingest import chunks_from_annotated_markdown
from rag.models import DocumentChunk, SearchHit
from rag.retriever import HybridRetriever
from rag.tool import execute_rag_search
from rag.versioning import MAX_TIMESTAMP, KnowledgeStatus, RetrievalFilter


_ROOT = pathlib.Path(__file__).parent.parent
_DEFAULT_CASES = _ROOT / "data" / "eval" / "enterprise_rag_cases.json"
_REPORTS_DIR = _ROOT / "data" / "eval" / "reports" / "enterprise_rag"
_DEFAULT_JSON_REPORT = _REPORTS_DIR / "enterprise_rag.json"
_DEFAULT_MD_REPORT = _REPORTS_DIR / "enterprise_rag.md"


@dataclass
class EnterpriseRAGCaseResult:
    case_id: str
    query: str
    answered: bool
    relevant_unit_ids: List[str]
    ranked_unit_ids: List[str]
    chunk_types: List[str]
    versions: List[str]
    parent_context_ids: List[str]
    recall_at_5: float
    reciprocal_rank: float
    chunk_type_hit: bool
    guardrail_hit: bool
    parent_context_hit: bool
    version_hit: bool
    metadata_boosted: int
    no_boost_reciprocal_rank: float
    metadata_boost_improved: bool
    rerank_fallback: bool
    temporal_hit: Optional[bool] = None


@dataclass
class EnterpriseRAGReport:
    generated_at: str
    dataset: str
    reproduce_command: str
    knowledge_files: List[str]
    chunk_count: int
    chunk_type_counts: Dict[str, int]
    version_counts: Dict[str, int]
    duplicate_chunk_ids: List[str]
    metadata_errors: List[str]
    parent_link_errors: List[str]
    case_count: int
    summary: Dict[str, float]
    cases: List[EnterpriseRAGCaseResult] = field(default_factory=list)


class DeterministicDenseIndex:
    """Stable lexical dense stand-in for deterministic local evaluation."""

    def __init__(self) -> None:
        self._index = BM25Index()

    def add(self, chunks: Sequence[DocumentChunk]) -> None:
        self._index.add(chunks)

    def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[RetrievalFilter] = None,
    ) -> List[SearchHit]:
        hits = self._index.search(query, top_k, filters=filters)
        return [
            hit.model_copy(update={
                "dense_score": hit.bm25_score,
                "bm25_score": None,
            })
            for hit in hits
        ]


class DeterministicReranker:
    def score(self, query: str, chunks: Sequence[DocumentChunk]) -> List[float]:
        query_terms = set(_tokens(query))
        scores = []
        for chunk in chunks:
            chunk_terms = set(_tokens(chunk.content))
            overlap = len(query_terms & chunk_terms)
            scores.append(overlap / (len(query_terms) + 1))
        return scores


class FailingReranker:
    def score(self, query: str, chunks: Sequence[DocumentChunk]) -> List[float]:
        del query, chunks
        raise RuntimeError("forced reranker failure")


def run_enterprise_rag_eval(
    *,
    dataset_path: pathlib.Path = _DEFAULT_CASES,
    top_k: int = 5,
) -> EnterpriseRAGReport:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    knowledge_files = [str(item) for item in payload["knowledge_files"]]
    chunks = _load_knowledge_chunks(knowledge_files)
    _apply_version_timeline(chunks)
    diagnostics = _ingest_diagnostics(chunks)

    retriever = _build_retriever(chunks, metadata_boost_enabled=True)
    no_boost = _build_retriever(chunks, metadata_boost_enabled=False)
    failing_reranker = _build_retriever(
        chunks,
        metadata_boost_enabled=True,
        reranker=FailingReranker(),
    )

    results = [
        _evaluate_case(case, retriever, no_boost, top_k)
        for case in payload["cases"]
    ]
    fallback_success = _evaluate_reranker_fallback(
        payload["cases"],
        failing_reranker,
        top_k,
    )
    summary = _summary(results)
    summary["rerank_fallback_success_rate"] = fallback_success

    try:
        dataset = str(dataset_path.relative_to(_ROOT))
    except ValueError:
        dataset = str(dataset_path)
    return EnterpriseRAGReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        dataset=dataset,
        reproduce_command=".venv/bin/python -m evaluation.enterprise_rag_eval",
        knowledge_files=knowledge_files,
        chunk_count=len(chunks),
        chunk_type_counts=diagnostics["chunk_type_counts"],
        version_counts=diagnostics["version_counts"],
        duplicate_chunk_ids=diagnostics["duplicate_chunk_ids"],
        metadata_errors=diagnostics["metadata_errors"],
        parent_link_errors=diagnostics["parent_link_errors"],
        case_count=len(results),
        summary=summary,
        cases=results,
    )


def _load_knowledge_chunks(paths: Sequence[str]) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    for value in paths:
        path = pathlib.Path(value)
        if not path.is_absolute():
            path = _ROOT / path
        chunks.extend(chunks_from_annotated_markdown(path.name, path.read_bytes()))
    return chunks


def _build_retriever(
    chunks: Sequence[DocumentChunk],
    *,
    metadata_boost_enabled: bool,
    reranker: Any = None,
) -> HybridRetriever:
    dense = DeterministicDenseIndex()
    sparse = BM25Index()
    dense.add(chunks)
    sparse.add(chunks)
    return HybridRetriever(
        dense,
        sparse,
        reranker=reranker if reranker is not None else DeterministicReranker(),
        relevance_threshold=0.0,
        metadata_boost_enabled=metadata_boost_enabled,
    )


def _evaluate_case(
    case: Dict[str, Any],
    retriever: HybridRetriever,
    no_boost: HybridRetriever,
    top_k: int,
) -> EnterpriseRAGCaseResult:
    params = {
        "query": case["query"],
        "top_k": top_k,
        "candidate_k": max(top_k * 4, 20),
        "mode": "rerank",
        "as_of": case.get("as_of"),
    }
    payload = execute_rag_search(retriever, params)
    no_boost_payload = execute_rag_search(no_boost, params)
    hits = payload.get("hits", [])
    relevant = set(case.get("relevant_unit_ids", []))
    ranked_units = [_unit_id(hit) for hit in hits]
    chunk_types = [_chunk_metadata(hit).get("chunk_type", "") for hit in hits]
    versions = [_chunk_metadata(hit).get("version", "") for hit in hits]
    parent_context_ids = [
        str(hit.get("parent_context", {}).get("parent_id"))
        for hit in hits
        if isinstance(hit, dict) and hit.get("parent_context")
    ]
    no_boost_ranked_units = [
        _unit_id(hit)
        for hit in no_boost_payload.get("hits", [])
    ]
    reciprocal_rank = _reciprocal_rank(relevant, ranked_units)
    no_boost_reciprocal_rank = _reciprocal_rank(relevant, no_boost_ranked_units)
    expected_types = set(case.get("expected_chunk_types", []))
    expected_parent_ids = set(case.get("expected_parent_ids", []))
    expected_version = case.get("expected_version")
    temporal = payload.get("temporal_comparison")
    temporal_hit = None
    if case.get("temporal_compare"):
        temporal_hit = bool(
            temporal
            and temporal.get("current_version") == expected_version
            and temporal.get("previous_version")
            == case.get("expected_previous_version")
        )

    return EnterpriseRAGCaseResult(
        case_id=case["id"],
        query=case["query"],
        answered=bool(payload.get("answered")),
        relevant_unit_ids=list(relevant),
        ranked_unit_ids=ranked_units,
        chunk_types=chunk_types,
        versions=versions,
        parent_context_ids=parent_context_ids,
        recall_at_5=_recall_at_k(relevant, ranked_units, top_k),
        reciprocal_rank=reciprocal_rank,
        chunk_type_hit=bool(expected_types & set(chunk_types)),
        guardrail_hit=(
            "guardrail" not in expected_types
            or "guardrail" in set(chunk_types)
        ),
        parent_context_hit=(
            not expected_parent_ids
            or bool(expected_parent_ids & set(parent_context_ids))
        ),
        version_hit=(
            not expected_version
            or expected_version in set(versions)
        ),
        metadata_boosted=int(payload.get("metadata", {}).get("metadata_boosted", 0)),
        no_boost_reciprocal_rank=no_boost_reciprocal_rank,
        metadata_boost_improved=reciprocal_rank > no_boost_reciprocal_rank,
        rerank_fallback=bool(payload.get("metadata", {}).get("rerank_fallback")),
        temporal_hit=temporal_hit,
    )


def _evaluate_reranker_fallback(
    cases: Sequence[Dict[str, Any]],
    retriever: HybridRetriever,
    top_k: int,
) -> float:
    values = []
    logger = logging.getLogger("rag.retriever")
    previous_level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        for case in cases:
            payload = execute_rag_search(
                retriever,
                {
                    "query": case["query"],
                    "top_k": top_k,
                    "candidate_k": max(top_k * 4, 20),
                    "mode": "rerank",
                    "as_of": case.get("as_of"),
                    "temporal_mode": "current",
                },
            )
            ranked = [_unit_id(hit) for hit in payload.get("hits", [])]
            relevant = set(case.get("relevant_unit_ids", []))
            values.append(
                bool(payload.get("metadata", {}).get("rerank_fallback"))
                and bool(relevant & set(ranked))
            )
    finally:
        logger.setLevel(previous_level)
    return _ratio(sum(values), len(values))


def _summary(results: Sequence[EnterpriseRAGCaseResult]) -> Dict[str, float]:
    temporal = [item for item in results if item.temporal_hit is not None]
    return {
        "recall_at_5": round(
            sum(item.recall_at_5 for item in results) / len(results),
            4,
        ) if results else 0.0,
        "mrr": round(
            sum(item.reciprocal_rank for item in results) / len(results),
            4,
        ) if results else 0.0,
        "chunk_type_recall": _rate(item.chunk_type_hit for item in results),
        "guardrail_recall": _rate(item.guardrail_hit for item in results),
        "parent_context_recall": _rate(item.parent_context_hit for item in results),
        "version_accuracy": _rate(item.version_hit for item in results),
        "temporal_comparison_accuracy": _rate(
            bool(item.temporal_hit) for item in temporal
        ),
        "metadata_boost_rate": _rate(
            item.metadata_boosted > 0 for item in results
        ),
        "metadata_boost_improvement_rate": _rate(
            item.metadata_boost_improved for item in results
        ),
    }


def _ingest_diagnostics(chunks: Sequence[DocumentChunk]) -> Dict[str, Any]:
    chunk_type_counts: Dict[str, int] = {}
    version_counts: Dict[str, int] = {}
    ids: Dict[str, int] = {}
    metadata_errors = []
    parent_link_errors = []
    parent_ids = {
        chunk.parent_id
        for chunk in chunks
        if chunk.metadata.get("chunk_type") == "section_parent"
    }
    for chunk in chunks:
        ids[chunk.chunk_id] = ids.get(chunk.chunk_id, 0) + 1
        chunk_type = str(chunk.metadata.get("chunk_type", ""))
        version = str(chunk.metadata.get("version", ""))
        chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1
        version_counts[version] = version_counts.get(version, 0) + 1
        for key in ("knowledge_id", "version", "version_id", "chunk_type", "unit_id"):
            if not chunk.metadata.get(key):
                metadata_errors.append(f"{chunk.chunk_id}: missing {key}")
        if (
            chunk.metadata.get("chunk_type") == "sop"
            and chunk.parent_id not in parent_ids
        ):
            parent_link_errors.append(
                f"{chunk.chunk_id}: missing parent {chunk.parent_id}",
            )
    return {
        "chunk_type_counts": dict(sorted(chunk_type_counts.items())),
        "version_counts": dict(sorted(version_counts.items())),
        "duplicate_chunk_ids": sorted(
            chunk_id for chunk_id, count in ids.items() if count > 1
        ),
        "metadata_errors": metadata_errors,
        "parent_link_errors": parent_link_errors,
    }


def _apply_version_timeline(chunks: Sequence[DocumentChunk]) -> None:
    grouped: Dict[str, Dict[str, List[DocumentChunk]]] = {}
    for chunk in chunks:
        knowledge_id = str(chunk.metadata.get("knowledge_id", ""))
        version_id = str(chunk.metadata.get("version_id", ""))
        grouped.setdefault(knowledge_id, {}).setdefault(version_id, []).append(chunk)
    for versions in grouped.values():
        published = [
            values
            for values in versions.values()
            if values
            and values[0].metadata.get("status") == KnowledgeStatus.PUBLISHED.value
        ]
        published.sort(key=lambda values: (
            float(values[0].metadata.get("effective_at", 0.0)),
            int(values[0].metadata.get("version_seq", 0)),
        ))
        for index, version_chunks in enumerate(published):
            next_effective = (
                float(published[index + 1][0].metadata["effective_at"])
                if index + 1 < len(published)
                else MAX_TIMESTAMP
            )
            for chunk in version_chunks:
                declared = float(
                    chunk.metadata.get(
                        "declared_expires_at",
                        chunk.metadata.get("expires_at", MAX_TIMESTAMP),
                    ),
                )
                chunk.metadata["expires_at"] = min(declared, next_effective)
                chunk.metadata["is_current"] = index == len(published) - 1


def _unit_id(hit: Dict[str, Any]) -> str:
    return str(_chunk_metadata(hit).get("unit_id", ""))


def _chunk_metadata(hit: Dict[str, Any]) -> Dict[str, Any]:
    chunk = hit.get("chunk") if isinstance(hit, dict) else {}
    chunk = chunk if isinstance(chunk, dict) else {}
    metadata = chunk.get("metadata") if isinstance(chunk, dict) else {}
    return metadata if isinstance(metadata, dict) else {}


def _tokens(text: str) -> List[str]:
    import re

    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9._-]+", text.lower())


def _recall_at_k(relevant: set[str], ranked: Sequence[str], k: int) -> float:
    return _ratio(len(relevant & set(ranked[:k])), len(relevant))


def _reciprocal_rank(relevant: set[str], ranked: Sequence[str]) -> float:
    for index, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def _rate(values: Any) -> float:
    items = list(values)
    return round(_ratio(sum(bool(item) for item in items), len(items)), 4)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def write_reports(
    report: EnterpriseRAGReport,
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
    lines = [
        "# GGBot Enterprise RAG Evaluation",
        "",
        f"- Dataset: `{report.dataset}`",
        f"- Chunks: {report.chunk_count}",
        f"- Cases: {report.case_count}",
        "",
        "## Summary",
        "",
    ]
    lines.extend(
        f"- {key}: {value:.4f}"
        for key, value in sorted(report.summary.items())
    )
    lines.extend([
        "",
        "## Diagnostics",
        "",
        f"- Chunk types: `{report.chunk_type_counts}`",
        f"- Versions: `{report.version_counts}`",
        f"- Duplicate chunk ids: {len(report.duplicate_chunk_ids)}",
        f"- Metadata errors: {len(report.metadata_errors)}",
        f"- Parent link errors: {len(report.parent_link_errors)}",
        "",
    ])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, default=_DEFAULT_CASES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    report = run_enterprise_rag_eval(dataset_path=args.dataset, top_k=args.top_k)
    if not args.no_write:
        write_reports(report)
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
