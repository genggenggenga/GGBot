"""Dense and sparse indexes used by hybrid retrieval."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Protocol, Sequence

from rag.models import DocumentChunk, SearchHit


class DenseIndex(Protocol):
    def add(self, chunks: Sequence[DocumentChunk]) -> None: ...
    def search(self, query: str, top_k: int) -> List[SearchHit]: ...


class ChromaDenseIndex:
    """Chroma-backed dense index with an injectable embedding function.

    Production callers can pass ``build_bge_embedding_function()``. Tests can
    inject a fake collection and never load a model or contact a server.
    """

    def __init__(
        self,
        collection: Optional[Any] = None,
        *,
        client: Optional[Any] = None,
        collection_name: str = "knowledge_dense",
        embedding_function: Optional[Any] = None,
    ) -> None:
        if collection is not None:
            self._collection = collection
            return
        if client is None:
            raise ValueError("collection or client is required")
        self._collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: Sequence[DocumentChunk]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            metadatas=[_chunk_metadata(chunk) for chunk in chunks],
        )

    def search(self, query: str, top_k: int) -> List[SearchHit]:
        result = self._collection.query(query_texts=[query], n_results=top_k)
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits: List[SearchHit] = []
        for chunk_id, content, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            score = max(0.0, 1.0 - float(distance))
            hits.append(SearchHit(
                chunk=_chunk_from_metadata(chunk_id, content, metadata or {}),
                score=score,
                dense_score=score,
            ))
        return hits


def build_bge_embedding_function(
    model_name: str = "BAAI/bge-small-zh-v1.5",
) -> Any:
    """Build Chroma's SentenceTransformer embedding function lazily."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    return SentenceTransformerEmbeddingFunction(model_name=model_name)


class BM25Index:
    """Small in-process BM25 index, independent from the vector store."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: List[DocumentChunk] = []
        self._tokens: List[List[str]] = []
        self._document_frequency: Counter[str] = Counter()
        self._average_length = 0.0

    def add(self, chunks: Sequence[DocumentChunk]) -> None:
        self._chunks = list(chunks)
        self._tokens = [tokenize(chunk.content) for chunk in self._chunks]
        self._document_frequency.clear()
        for tokens in self._tokens:
            self._document_frequency.update(set(tokens))
        total = sum(len(tokens) for tokens in self._tokens)
        self._average_length = total / len(self._tokens) if self._tokens else 0.0

    def search(self, query: str, top_k: int) -> List[SearchHit]:
        query_tokens = tokenize(query)
        if not query_tokens or not self._chunks:
            return []
        scores = [
            self._score(query_tokens, tokens)
            for tokens in self._tokens
        ]
        ranked = sorted(
            enumerate(scores), key=lambda item: item[1], reverse=True
        )
        return [
            SearchHit(
                chunk=self._chunks[index],
                score=score,
                bm25_score=score,
            )
            for index, score in ranked[:top_k]
            if score > 0
        ]

    def _score(self, query: List[str], document: List[str]) -> float:
        frequencies = Counter(document)
        document_length = len(document)
        document_count = len(self._tokens)
        score = 0.0
        for token in query:
            frequency = frequencies[token]
            if not frequency:
                continue
            df = self._document_frequency[token]
            idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
            normalizer = frequency + self.k1 * (
                1 - self.b + self.b * document_length / (self._average_length or 1)
            )
            score += idf * frequency * (self.k1 + 1) / normalizer
        return score


def tokenize(text: str) -> List[str]:
    normalized = text.lower()
    latin = re.findall(r"[a-z0-9][a-z0-9._#-]*", normalized)
    chinese = re.findall(r"[\u4e00-\u9fff]", normalized)
    return latin + chinese


def _chunk_metadata(chunk: DocumentChunk) -> Dict[str, Any]:
    return {
        "source": chunk.source,
        "title": chunk.title,
        "section": chunk.section,
        "page": chunk.page or 0,
        "chunk_index": chunk.chunk_index,
        "parent_id": chunk.parent_id,
        **chunk.metadata,
    }


def _chunk_from_metadata(
    chunk_id: str,
    content: str,
    metadata: Dict[str, Any],
) -> DocumentChunk:
    page = int(metadata.get("page", 0)) or None
    reserved = {"source", "title", "section", "page", "chunk_index", "parent_id"}
    return DocumentChunk(
        chunk_id=chunk_id,
        content=content,
        source=str(metadata.get("source", "")),
        title=str(metadata.get("title", "")),
        section=str(metadata.get("section", "")),
        page=page,
        chunk_index=int(metadata.get("chunk_index", 0)),
        parent_id=str(metadata.get("parent_id", "")),
        metadata={key: value for key, value in metadata.items() if key not in reserved},
    )
