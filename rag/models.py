"""Shared data models for the RAG pipeline."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class LoadedSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    source: str
    title: str = ""
    section: str = ""
    page: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    content: str
    source: str
    title: str = ""
    section: str = ""
    page: Optional[int] = None
    chunk_index: int = Field(ge=0)
    parent_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk: DocumentChunk
    score: float = 0.0
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_id: str
    chunk_id: str
    source: str
    title: str = ""
    section: str = ""
    page: Optional[int] = None
    knowledge_id: Optional[str] = None
    version: Optional[str] = None
    effective_at: Optional[float] = None
    expires_at: Optional[float] = None


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    hits: List[SearchHit] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    answered: bool
    reason: Optional[str] = None
    queries: List[str] = Field(default_factory=list)


class QueryPlan(BaseModel):
    """Validated output of reference resolution and multi-query rewriting."""

    model_config = ConfigDict(extra="forbid")

    original_query: str
    standalone_query: str
    alternative_queries: List[str] = Field(default_factory=list)
    resolved_references: Dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    used_llm: bool = False
    fallback_reason: Optional[str] = None

    def retrieval_queries(self, max_queries: int = 3) -> List[str]:
        values = [
            self.original_query,
            self.standalone_query,
            *self.alternative_queries,
        ]
        queries = []
        for value in values:
            normalized = value.strip()
            if normalized and normalized not in queries:
                queries.append(normalized)
        return queries[:max_queries]
