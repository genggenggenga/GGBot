"""ToolRegistry integration for hybrid RAG."""
from typing import Iterable, Optional

from core.tool_registry import LocalToolAdapter, ToolRegistry, ToolSpec, ToolType
from rag.retriever import HybridRetriever


def register_rag_tool(
    registry: ToolRegistry,
    retriever: HybridRetriever,
    *,
    agent_names: Optional[Iterable[str]] = None,
) -> None:
    async def search_handler(params, context):
        mode = params.get("mode", "rerank")
        result = retriever.search(
            params["query"],
            top_k=params.get("top_k", 5),
            candidate_k=params.get("candidate_k", 20),
            use_sparse=mode != "dense",
            use_reranker=mode == "rerank",
        )
        return result.model_dump(mode="json")

    spec = ToolSpec(
        name="rag_search",
        description="Hybrid knowledge search with citations",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
                "candidate_k": {"type": "integer"},
                "mode": {
                    "type": "string",
                    "enum": ["dense", "hybrid", "rerank"],
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
