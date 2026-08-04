"""Runtime assembly and synchronized ingestion for hybrid RAG."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from rag.indexes import BM25Index, ChromaDenseIndex, build_bge_embedding_function
from rag.loaders import chunk_sections, load_document
from rag.models import DocumentChunk, LoadedSection
from rag.retriever import CrossEncoderReranker, HybridRetriever


class KnowledgeRuntime:
    """Keep the dense and sparse indexes synchronized during ingestion."""

    def __init__(
        self,
        knowledge_base: Any,
        dense_index: ChromaDenseIndex,
        sparse_index: BM25Index,
        retriever: HybridRetriever,
        chunks: Sequence[DocumentChunk] = (),
        write_dense_on_ingest: bool = True,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.retriever = retriever
        self._write_dense_on_ingest = write_dense_on_ingest
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}

    @classmethod
    def build(
        cls,
        knowledge_base: Any,
        *,
        enable_local_models: bool = True,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        relevance_threshold: float = 0.0,
    ) -> "KnowledgeRuntime":
        chunks = _collection_chunks(knowledge_base._collection)
        if enable_local_models:
            embedding_function = build_bge_embedding_function(embedding_model)
            dense_index = ChromaDenseIndex(
                client=knowledge_base._client,
                collection_name="knowledge_dense_bge",
                embedding_function=embedding_function,
            )
            reranker = CrossEncoderReranker(reranker_model)
            dense_index.add(chunks)
        else:
            dense_index = ChromaDenseIndex(collection=knowledge_base._collection)
            reranker = None

        sparse_index = BM25Index()
        sparse_index.add(chunks)
        retriever = HybridRetriever(
            dense_index,
            sparse_index,
            reranker=reranker,
            relevance_threshold=relevance_threshold,
        )
        return cls(
            knowledge_base,
            dense_index,
            sparse_index,
            retriever,
            chunks,
            write_dense_on_ingest=enable_local_models,
        )

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        sections = [
            LoadedSection(
                text=document.get("content", ""),
                source=document.get("source") or document.get("title", "inline"),
                title=document.get("title", ""),
                section=document.get("section", ""),
            )
            for document in documents
            if document.get("content", "").strip()
        ]
        return self.add_sections(sections)

    def add_file(self, filename: str, content: bytes) -> int:
        suffix = Path(filename).suffix.lower()
        if suffix not in {".txt", ".md", ".markdown", ".pdf"}:
            raise ValueError(f"unsupported document type: {suffix}")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / Path(filename).name
            path.write_bytes(content)
            sections = [
                section.model_copy(update={"source": filename})
                for section in load_document(path)
            ]
            return self.add_sections(sections)

    def add_sections(self, sections: Iterable[LoadedSection]) -> int:
        loaded = list(sections)
        chunks = chunk_sections(loaded)
        if not chunks:
            return 0

        self.knowledge_base.add_documents([
            {
                "title": section.title,
                "content": section.text,
                "source": section.source,
                "section": section.section,
            }
            for section in loaded
        ])
        if self._write_dense_on_ingest:
            self.dense_index.add(chunks)
        self._chunks.update({chunk.chunk_id: chunk for chunk in chunks})
        self.sparse_index.add(list(self._chunks.values()))
        return len(chunks)


def local_models_enabled() -> bool:
    value = os.getenv("RAG_LOCAL_MODELS_ENABLED", "true")
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _collection_chunks(collection: Any) -> List[DocumentChunk]:
    data = collection.get(include=["documents", "metadatas"])
    chunks = []
    for index, (chunk_id, content, metadata) in enumerate(zip(
        data.get("ids", []),
        data.get("documents", []),
        data.get("metadatas", []),
    )):
        meta = metadata or {}
        chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            content=content,
            source=str(meta.get("source") or meta.get("title") or "knowledge_base"),
            title=str(meta.get("title", "")),
            section=str(meta.get("section", "")),
            page=int(meta.get("page") or 0) or None,
            chunk_index=int(meta.get("chunk_index", index)),
            parent_id=str(meta.get("parent_id") or f"document-{chunk_id}"),
        ))
    return chunks
