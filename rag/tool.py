"""ToolRegistry integration for hybrid RAG."""
import asyncio
import re
from typing import Any, Dict, Iterable, Optional, Sequence

from core.tool_registry import LocalToolAdapter, ToolRegistry, ToolSpec, ToolType
from rag.retriever import HybridRetriever
from rag.versioning import RetrievalFilter


_TEMPORAL_CHANGE_PATTERN = re.compile(
    r"(?:最近|近期|之前|以前|历史|新旧|变化|变更|调整|更新|改了|区别|相比)",
    re.IGNORECASE,
)


def register_rag_tool(
    registry: ToolRegistry,
    retriever: HybridRetriever,
    *,
    agent_names: Optional[Iterable[str]] = None,
) -> None:
    async def search_handler(params, context):
        del context
        mode = params.get("mode", "rerank")
        queries = params.get("queries") or [params["query"]]
        current_filter = RetrievalFilter.current(as_of=params.get("as_of"))
        result = await asyncio.to_thread(
            _search,
            retriever,
            queries,
            params["query"],
            params.get("top_k", 5),
            params.get("candidate_k", 20),
            mode,
            current_filter,
        )
        temporal_mode = _resolve_temporal_mode(
            params.get("temporal_mode", "auto"),
            params["query"],
        )
        if temporal_mode != "compare_previous" or not result.answered:
            payload = result.model_dump(mode="json")
            payload["temporal_mode"] = temporal_mode
            return payload

        current_hit = result.hits[0]
        knowledge_id = current_hit.chunk.metadata.get("knowledge_id")
        effective_at = current_hit.chunk.metadata.get("effective_at")
        previous = None
        if knowledge_id and isinstance(effective_at, (int, float)):
            previous_filter = RetrievalFilter.current(
                as_of=float(effective_at) - 0.000001,
                knowledge_ids=[str(knowledge_id)],
            )
            previous = await asyncio.to_thread(
                _search,
                retriever,
                queries,
                params["query"],
                params.get("top_k", 5),
                params.get("candidate_k", 20),
                mode,
                previous_filter,
            )
        return _temporal_payload(result, previous, str(knowledge_id or ""))

    spec = ToolSpec(
        name="rag_search",
        description="Hybrid knowledge search with citations",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "as_of": {
                    "type": ["string", "number"],
                },
                "top_k": {"type": "integer"},
                "candidate_k": {"type": "integer"},
                "mode": {
                    "type": "string",
                    "enum": ["dense", "hybrid", "rerank"],
                },
                "temporal_mode": {
                    "type": "string",
                    "enum": ["auto", "current", "compare_previous"],
                },
            },
            "required": ["query"],
        },
        output_schema={"type": "object"},
        tool_type=ToolType.READ,
        timeout_s=30.0,
        cache_ttl=300.0,
    )
    registry.register(LocalToolAdapter(spec, search_handler))
    for agent_name in agent_names or (
        "knowledge",
        "order",
        "logistics",
        "after_sales",
    ):
        registry.add_to_agent_whitelist(agent_name, spec.name)


def _search(
    retriever: Any,
    queries: Sequence[str],
    rerank_query: str,
    top_k: int,
    candidate_k: int,
    mode: str,
    filters: RetrievalFilter,
):
    search_multi = getattr(retriever, "search_multi", None)
    if search_multi is not None:
        return search_multi(
            queries,
            rerank_query=rerank_query,
            top_k=top_k,
            candidate_k=candidate_k,
            use_sparse=mode != "dense",
            use_reranker=mode == "rerank",
            filters=filters,
        )
    try:
        return retriever.search(
            rerank_query,
            top_k=top_k,
            candidate_k=candidate_k,
            use_sparse=mode != "dense",
            use_reranker=mode == "rerank",
            filters=filters,
        )
    except TypeError as ex:
        if "unexpected keyword argument" not in str(ex):
            raise
        return retriever.search(
            rerank_query,
            top_k=top_k,
            candidate_k=candidate_k,
            use_sparse=mode != "dense",
            use_reranker=mode == "rerank",
        )


def _resolve_temporal_mode(mode: str, query: str) -> str:
    if mode == "auto":
        return (
            "compare_previous"
            if _TEMPORAL_CHANGE_PATTERN.search(query)
            else "current"
        )
    return mode


def _temporal_payload(current, previous, knowledge_id: str) -> Dict[str, Any]:
    current_data = current.model_dump(mode="json")
    current_text = _version_text(current.hits, knowledge_id)
    previous_answered = previous is not None and previous.answered
    previous_text = (
        _version_text(previous.hits, knowledge_id)
        if previous_answered
        else ""
    )
    current_units = _text_units(current_text)
    previous_units = _text_units(previous_text)
    added = [item for item in current_units if item not in previous_units]
    removed = [item for item in previous_units if item not in current_units]

    citations = []
    for source in (
        current.citations,
        previous.citations if previous_answered else [],
    ):
        for citation in source:
            item = citation.model_dump(mode="json")
            item["citation_id"] = f"[{len(citations) + 1}]"
            citations.append(item)
    current_version = current.hits[0].chunk.metadata.get("version")
    previous_version = (
        previous.hits[0].chunk.metadata.get("version")
        if previous_answered and previous.hits
        else None
    )
    current_data.update({
        "citations": citations,
        "temporal_mode": "compare_previous",
        "temporal_comparison": {
            "knowledge_id": knowledge_id or None,
            "current_version": current_version,
            "previous_version": previous_version,
            "previous_found": bool(previous_answered),
            "changed": bool(previous_answered and (added or removed)),
            "added": added[:8],
            "removed": removed[:8],
        },
    })
    return current_data


def _version_text(hits: Sequence[Any], knowledge_id: str) -> str:
    values = []
    for hit in hits:
        if (
            knowledge_id
            and hit.chunk.metadata.get("knowledge_id") != knowledge_id
        ):
            continue
        if hit.chunk.content not in values:
            values.append(hit.chunk.content)
    return "\n".join(values)


def _text_units(text: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"(?<=[。！？!?])|\n+", text)
        if item.strip()
    ]
