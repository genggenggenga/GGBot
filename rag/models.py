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


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    hits: List[SearchHit] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    answered: bool
    reason: Optional[str] = None
