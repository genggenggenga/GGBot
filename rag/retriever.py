"""Hybrid retrieval, reciprocal-rank fusion, reranking, and citations."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Protocol, Sequence

import httpx

from rag.indexes import DenseIndex
from rag.models import Citation, DocumentChunk, RetrievalResult, SearchHit


class SparseIndex(Protocol):
    def add(self, chunks: Sequence[DocumentChunk]) -> None: ...
    def search(self, query: str, top_k: int) -> List[SearchHit]: ...


class Reranker(Protocol):
    def score(self, query: str, chunks: Sequence[DocumentChunk]) -> List[float]: ...


class CrossEncoderReranker:
    """Lazy Cross-Encoder wrapper; callers may inject a fake in tests."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        model: Optional[object] = None,
    ) -> None:
        self._model_name = model_name
        self._model = model

    def score(self, query: str, chunks: Sequence[DocumentChunk]) -> List[float]:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        pairs = [(query, chunk.content) for chunk in chunks]
        values = self._model.predict(pairs)  # type: ignore[attr-defined]
        return [float(value) for value in values]


class SiliconFlowReranker:
    """Reranker that delegates to the SiliconFlow ``/v1/rerank`` API.

    Implements the same ``score`` contract as ``CrossEncoderReranker`` so it
    can be used as a drop-in replacement without pulling in ``torch``.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://api.siliconflow.cn/v1",
        timeout: float = 30.0,
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key or os.getenv("SILICONFLOW_API_KEY", "")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        if not self._api_key:
            raise ValueError("SiliconFlowReranker requires SILICONFLOW_API_KEY")

    def score(self, query: str, chunks: Sequence[DocumentChunk]) -> List[float]:
        documents = [chunk.content for chunk in chunks]
        if not documents:
            return []
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/rerank",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model_name,
                    "query": query,
                    "documents": documents,
                    "return_documents": False,
                },
            )
        if response.status_code != 200:
            raise RuntimeError(
                f"SiliconFlow rerank API returned "
                f"{response.status_code}: {response.text[:200]}"
            )
        payload = response.json()
        # Map API results back to the original chunk order.
        scores = [0.0] * len(chunks)
        for item in payload.get("results", []):
            index = int(item["index"])
            if 0 <= index < len(scores):
                scores[index] = float(item["relevance_score"])
        return scores


class HybridRetriever:
    def __init__(
        self,
        dense_index: DenseIndex,
        sparse_index: SparseIndex,
        *,
        reranker: Optional[Reranker] = None,
        rrf_k: int = 60,
        relevance_threshold: float = 0.0,
    ) -> None:
        self._dense = dense_index
        self._sparse = sparse_index
        self._reranker = reranker
        self._rrf_k = rrf_k
        self._threshold = relevance_threshold

    def add(self, chunks: Sequence[DocumentChunk]) -> None:
        self._dense.add(chunks)
        self._sparse.add(chunks)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidate_k: int = 20,
        use_sparse: bool = True,
        use_reranker: bool = True,
    ) -> RetrievalResult:
        dense_hits = self._dense.search(query, candidate_k)
        if use_sparse:
            sparse_hits = self._sparse.search(query, candidate_k)
            candidates = reciprocal_rank_fusion(
                dense_hits, sparse_hits, rrf_k=self._rrf_k
            )
        else:
            candidates = dense_hits

        if use_reranker and self._reranker and candidates:
            scores = self._reranker.score(
                query, [hit.chunk for hit in candidates]
            )
            for hit, score in zip(candidates, scores):
                hit.rerank_score = score
                hit.score = score
            candidates.sort(key=lambda hit: hit.score, reverse=True)

        selected = candidates[:top_k]
        if not selected or selected[0].score < self._threshold:
            return RetrievalResult(
                query=query,
                answered=False,
                reason="no_relevant_evidence",
            )

        citations = [
            Citation(
                citation_id=f"[{index}]",
                chunk_id=hit.chunk.chunk_id,
                source=hit.chunk.source,
                title=hit.chunk.title,
                section=hit.chunk.section,
                page=hit.chunk.page,
            )
            for index, hit in enumerate(selected, start=1)
        ]
        return RetrievalResult(
            query=query,
            hits=selected,
            citations=citations,
            answered=True,
        )


def reciprocal_rank_fusion(
    dense_hits: Sequence[SearchHit],
    sparse_hits: Sequence[SearchHit],
    *,
    rrf_k: int = 60,
) -> List[SearchHit]:
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    merged: Dict[str, SearchHit] = {}
    scores: Dict[str, float] = {}
    for hits, score_field in (
        (dense_hits, "dense_score"),
        (sparse_hits, "bm25_score"),
    ):
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk.chunk_id
            if chunk_id not in merged:
                merged[chunk_id] = hit.model_copy(deep=True)
            existing = merged[chunk_id]
            source_score = getattr(hit, score_field)
            if source_score is not None:
                setattr(existing, score_field, source_score)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    for chunk_id, hit in merged.items():
        hit.rrf_score = scores[chunk_id]
        hit.score = scores[chunk_id]
    return sorted(merged.values(), key=lambda hit: hit.score, reverse=True)
