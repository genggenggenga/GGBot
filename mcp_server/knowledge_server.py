"""Knowledge MCP server exposing the shared hybrid RAG contract."""

from functools import lru_cache
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from mcp.knowledge_base import KnowledgeBase
from rag.runtime import KnowledgeRuntime, local_models_enabled
from rag.tool import execute_rag_search


server = FastMCP(
    "ggbot-knowledge",
    instructions="Hybrid knowledge retrieval with citations and version comparison.",
)


@lru_cache(maxsize=1)
def _runtime() -> KnowledgeRuntime:
    knowledge_base = KnowledgeBase(
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv(
            "CHROMA_PERSIST_DIRECTORY",
            "/app/data/chroma",
        ),
    )
    legacy_threshold = os.getenv("RAG_RELEVANCE_THRESHOLD")
    return KnowledgeRuntime.build(
        knowledge_base,
        enable_local_models=local_models_enabled(),
        embedding_provider=os.getenv("RAG_EMBEDDING_PROVIDER"),
        embedding_model=os.getenv(
            "RAG_EMBEDDING_MODEL",
            "BAAI/bge-small-zh-v1.5",
        ),
        reranker_model=os.getenv(
            "RAG_RERANKER_MODEL",
            "BAAI/bge-reranker-v2-m3",
        ),
        relevance_threshold=(
            float(legacy_threshold)
            if legacy_threshold not in {None, ""}
            else None
        ),
        dense_threshold=float(os.getenv("RAG_DENSE_THRESHOLD", "0.2")),
        rrf_threshold=float(os.getenv("RAG_RRF_THRESHOLD", "0.01")),
        rerank_threshold=float(os.getenv("RAG_RERANK_THRESHOLD", "0.1")),
    )


@server.tool()
def rag_search(
    query: str,
    queries: Optional[List[str]] = None,
    as_of: Optional[str] = None,
    top_k: int = 5,
    candidate_k: int = 20,
    mode: str = "rerank",
    temporal_mode: str = "auto",
) -> Dict[str, Any]:
    """Search current knowledge and optionally compare the previous version."""
    return execute_rag_search(_runtime().retriever, {
        "query": query,
        "queries": queries,
        "as_of": as_of,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "mode": mode,
        "temporal_mode": temporal_mode,
    })


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
