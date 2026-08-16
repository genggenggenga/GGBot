"""Hybrid retrieval, reciprocal-rank fusion, reranking, and citations."""
from __future__ import annotations

import os
import re
import logging
from typing import Any, Dict, List, Optional, Protocol, Sequence

import httpx

from rag.indexes import DenseIndex
from rag.models import Citation, DocumentChunk, RetrievalResult, SearchHit
from rag.versioning import RetrievalFilter, is_metadata_visible


logger = logging.getLogger(__name__)


class SparseIndex(Protocol):
    def add(self, chunks: Sequence[DocumentChunk]) -> None: ...
    def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[RetrievalFilter] = None,
    ) -> List[SearchHit]: ...


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
        relevance_threshold: Optional[float] = None,
        dense_threshold: float = 0.2,
        rrf_threshold: float = 0.01,
        rerank_threshold: float = 0.1,
        metadata_boost_enabled: bool = True,
    ) -> None:
        self._dense = dense_index
        self._sparse = sparse_index
        self._reranker = reranker
        self._rrf_k = rrf_k
        if relevance_threshold is not None:
            dense_threshold = rrf_threshold = rerank_threshold = (
                relevance_threshold
            )
        self._dense_threshold = dense_threshold
        self._rrf_threshold = rrf_threshold
        self._rerank_threshold = rerank_threshold
        self._metadata_boost_enabled = metadata_boost_enabled

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
        filters: Optional[RetrievalFilter] = None,
    ) -> RetrievalResult:
        return self.search_multi(
            [query],
            rerank_query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            use_sparse=use_sparse,
            use_reranker=use_reranker,
            filters=filters,
        )

    def search_multi(
        self,
        queries: Sequence[str],
        *,
        rerank_query: Optional[str] = None,
        top_k: int = 5,
        candidate_k: int = 20,
        use_sparse: bool = True,
        use_reranker: bool = True,
        filters: Optional[RetrievalFilter] = None,
    ) -> RetrievalResult:
        unique_queries = list(dict.fromkeys(
            query.strip() for query in queries if query.strip()
        ))
        if not unique_queries:
            return RetrievalResult(
                query="",
                answered=False,
                reason="empty_query",
            )
        diagnostics: Dict[str, Any] = {
            "rerank_fallback": False,
            "metadata_boosted": 0,
            "metadata_boost_total": 0.0,
        }
        active_filter = filters or RetrievalFilter.current()
        ranked_lists: List[Sequence[SearchHit]] = []
        for query in unique_queries:
            ranked_lists.append(self._search_index(
                self._dense,
                query,
                candidate_k,
                active_filter,
            ))
            if use_sparse:
                ranked_lists.append(self._search_index(
                    self._sparse,
                    query,
                    candidate_k,
                    active_filter,
                ))
        if len(ranked_lists) == 1:
            candidates = list(ranked_lists[0])
        else:
            candidates = reciprocal_rank_fusion_many(
                ranked_lists,
                rrf_k=self._rrf_k,
            )

        ranking_query = rerank_query or unique_queries[0]
        if self._metadata_boost_enabled:
            diagnostics.update(_apply_metadata_boost(ranking_query, candidates))
        candidates.sort(key=lambda hit: hit.score, reverse=True)

        reranked = False
        if use_reranker and self._reranker and candidates:
            try:
                scores = self._reranker.score(
                    ranking_query,
                    [hit.chunk for hit in candidates],
                )
                if len(scores) != len(candidates):
                    raise RuntimeError("reranker returned mismatched score count")
                for hit, score in zip(candidates, scores):
                    hit.rerank_score = score
                    hit.score = score + (hit.metadata_boost_score or 0.0)
                candidates.sort(key=lambda hit: hit.score, reverse=True)
                reranked = True
            except Exception as ex:
                diagnostics["rerank_fallback"] = True
                diagnostics["rerank_fallback_reason"] = type(ex).__name__
                logger.warning(
                    "RAG reranker failed, falling back to fused ranking: %s",
                    ex,
                )

        selected = candidates[:top_k]
        threshold = (
            self._rerank_threshold
            if reranked
            else self._rrf_threshold if use_sparse else self._dense_threshold
        )
        if not selected or selected[0].score < threshold:
            return RetrievalResult(
                query=rerank_query or unique_queries[0],
                answered=False,
                reason="no_relevant_evidence",
                queries=unique_queries,
                metadata=diagnostics,
            )

        citations = [
            Citation(
                citation_id=f"[{index}]",
                chunk_id=hit.chunk.chunk_id,
                source=hit.chunk.source,
                title=hit.chunk.title,
                section=hit.chunk.section,
                page=hit.chunk.page,
                knowledge_id=hit.chunk.metadata.get("knowledge_id"),
                version=hit.chunk.metadata.get("version"),
                effective_at=hit.chunk.metadata.get("effective_at"),
                expires_at=hit.chunk.metadata.get("expires_at"),
            )
            for index, hit in enumerate(selected, start=1)
        ]
        return RetrievalResult(
            query=rerank_query or unique_queries[0],
            hits=selected,
            citations=citations,
            answered=True,
            queries=unique_queries,
            metadata=diagnostics,
        )

    def parent_contexts(
        self,
        hits: Sequence[SearchHit],
        *,
        filters: Optional[RetrievalFilter] = None,
    ) -> Dict[str, DocumentChunk]:
        parent_ids = {
            hit.chunk.parent_id
            for hit in hits
            if hit.chunk.parent_id
            and hit.chunk.metadata.get("chunk_type") != "section_parent"
        }
        if not parent_ids:
            return {}
        chunks = getattr(self._sparse, "_chunks", [])
        active_filter = filters or RetrievalFilter.current()
        parents: Dict[str, DocumentChunk] = {}
        for chunk in chunks:
            if chunk.parent_id not in parent_ids:
                continue
            if chunk.metadata.get("chunk_type") != "section_parent":
                continue
            if not is_metadata_visible(
                {**chunk.metadata, "source": chunk.source, "title": chunk.title},
                active_filter,
            ):
                continue
            parents[chunk.parent_id] = chunk
        return parents

    @staticmethod
    def _search_index(
        index: Any,
        query: str,
        top_k: int,
        filters: RetrievalFilter,
    ) -> List[SearchHit]:
        try:
            return index.search(query, top_k, filters=filters)
        except TypeError as ex:
            if "unexpected keyword argument" not in str(ex):
                raise
            return index.search(query, top_k)


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


def reciprocal_rank_fusion_many(
    ranked_lists: Sequence[Sequence[SearchHit]],
    *,
    rrf_k: int = 60,
) -> List[SearchHit]:
    """Fuse any number of dense/sparse/query result lists by chunk ID."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    merged: Dict[str, SearchHit] = {}
    scores: Dict[str, float] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk.chunk_id
            if chunk_id not in merged:
                merged[chunk_id] = hit.model_copy(deep=True)
            existing = merged[chunk_id]
            if hit.dense_score is not None:
                existing.dense_score = hit.dense_score
            if hit.bm25_score is not None:
                existing.bm25_score = hit.bm25_score
            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + 1.0 / (rrf_k + rank)
            )
    for chunk_id, hit in merged.items():
        hit.rrf_score = scores[chunk_id]
        hit.score = scores[chunk_id]
    return sorted(merged.values(), key=lambda hit: hit.score, reverse=True)


_INTENT_KEYWORDS = {
    "refund_request": ("退款", "退货", "售后", "退回", "仅退款"),
    "logistics_delay": ("物流", "延迟", "没收到", "配送", "快递", "补偿"),
    "coupon_issue": ("优惠券", "券", "过期", "补发"),
    "invoice_request": ("发票", "开票", "税号", "抬头"),
    "privacy_protection": ("隐私", "手机号", "身份证", "地址", "别人", "他人"),
    "write_action_confirmation": (
        "直接退",
        "帮我退",
        "取消",
        "改地址",
        "补发",
        "赔付",
        "执行",
    ),
    "human_handoff": ("人工", "投诉", "升级", "客服", "工单"),
}
_DOMAIN_KEYWORDS = {
    "after_sales": ("退款", "退货", "售后", "换货"),
    "logistics": ("物流", "快递", "配送", "签收"),
    "membership": ("会员", "积分", "优惠券"),
    "order_payment": ("订单", "支付", "发票"),
    "safety": ("隐私", "确认", "越权", "安全", "密码"),
}
_GUARDRAIL_PATTERN = re.compile(
    r"(直接|立刻|马上|确认|取消|退款|改地址|补发|赔付|手机号|身份证|他人|别人|隐私|密码)"
)
_TABLE_PATTERN = re.compile(r"(标准|多少|金额|补偿|费用|时效|等级|规则)")


def _apply_metadata_boost(
    query: str,
    hits: Sequence[SearchHit],
) -> Dict[str, Any]:
    boosted = 0
    total = 0.0
    for hit in hits:
        boost = _metadata_boost(query, hit.chunk)
        hit.metadata_boost_score = boost if boost else None
        if boost:
            hit.score += boost
            boosted += 1
            total += boost
    return {
        "metadata_boosted": boosted,
        "metadata_boost_total": round(total, 6),
    }


def _metadata_boost(query: str, chunk: DocumentChunk) -> float:
    text = query.lower()
    metadata = chunk.metadata
    chunk_type = str(metadata.get("chunk_type", "")).lower()
    boost = 0.0

    if chunk_type == "section_parent":
        boost -= 0.02
    if (
        chunk_type == "guardrail"
        or metadata.get("guardrail") is True
        or str(metadata.get("risk_level", "")).lower() == "high"
    ) and _GUARDRAIL_PATTERN.search(query):
        boost += 0.15
    intent = str(metadata.get("intent", "")).lower()
    if any(keyword in query for keyword in _INTENT_KEYWORDS.get(intent, ())):
        boost += 0.08
    domain = str(metadata.get("business_domain", "")).lower()
    if any(keyword in query for keyword in _DOMAIN_KEYWORDS.get(domain, ())):
        boost += 0.04
    if chunk_type == "table" and _TABLE_PATTERN.search(query):
        boost += 0.04
    if chunk_type == "policy_rule" and any(
        keyword in query for keyword in ("规则", "政策", "能不能", "是否", "不支持")
    ):
        boost += 0.03
    if intent and intent.replace("_", " ") in text:
        boost += 0.03
    return round(boost, 6)
