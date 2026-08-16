"""Runtime assembly and synchronized ingestion for hybrid RAG."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from rag.indexes import (
    BM25Index,
    ChromaDenseIndex,
)
from rag.loaders import ChunkingConfig, chunk_sections, load_document
from rag.markdown_ingest import chunks_from_annotated_markdown
from rag.models import DocumentChunk, LoadedSection
from rag.retriever import (
    CrossEncoderReranker,
    HybridRetriever,
    Reranker,
    SiliconFlowReranker,
)
from rag.versioning import normalize_knowledge_metadata


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
        chunking_config: Optional[ChunkingConfig] = None,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.retriever = retriever
        self._write_dense_on_ingest = write_dense_on_ingest
        self._chunking_config = chunking_config or ChunkingConfig.from_env()
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}

    @classmethod
    def build(
        cls,
        knowledge_base: Any,
        *,
        enable_local_models: bool = True,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        relevance_threshold: Optional[float] = None,
        dense_threshold: float = 0.2,
        rrf_threshold: float = 0.01,
        rerank_threshold: float = 0.1,
        embedding_provider: Optional[str] = None,
    ) -> "KnowledgeRuntime":
        """Assemble the runtime.

        Provider selection (in priority order):

        1. ``embedding_provider`` argument — ``"api"`` / ``"local"`` / ``"off"``
        2. environment variable ``RAG_EMBEDDING_PROVIDER``
        3. legacy ``enable_local_models`` boolean (kept for backward compat)

        - ``local``: load BGE + CrossEncoder in-process (needs torch)
        - ``api``:   delegate to SiliconFlow API (needs SILICONFLOW_API_KEY)
        - ``off``:   no embedding / no reranker, BM25-only retrieval
        """
        provider = (embedding_provider or _embedding_provider() or "").lower()
        if provider in {"", "auto"}:
            provider = "local" if enable_local_models else "off"

        chunks = _collection_chunks(knowledge_base._collection)
        if provider == "api":
            dense_index = ChromaDenseIndex(
                collection=knowledge_base._collection,
            )
            reranker: Optional[Reranker] = SiliconFlowReranker(reranker_model)
            write_dense = False
        elif provider == "local":
            dense_index = ChromaDenseIndex(
                collection=knowledge_base._collection,
            )
            reranker = CrossEncoderReranker(reranker_model)
            write_dense = False
        else:  # "off"
            dense_index = ChromaDenseIndex(collection=knowledge_base._collection)
            reranker = None
            write_dense = False

        sparse_index = BM25Index()
        sparse_index.add(chunks)
        retriever = HybridRetriever(
            dense_index,
            sparse_index,
            reranker=reranker,
            relevance_threshold=relevance_threshold,
            dense_threshold=dense_threshold,
            rrf_threshold=rrf_threshold,
            rerank_threshold=rerank_threshold,
        )
        return cls(
            knowledge_base,
            dense_index,
            sparse_index,
            retriever,
            chunks,
            write_dense_on_ingest=write_dense,
        )

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        sections = []
        version_fields = {
            "knowledge_id",
            "version_id",
            "version",
            "version_seq",
            "status",
            "effective_at",
            "expires_at",
            "is_current",
        }
        for document in documents:
            content = document.get("content", "")
            if not content.strip():
                continue
            metadata = dict(document.get("metadata", {}))
            metadata.update({
                key: document[key]
                for key in version_fields
                if document.get(key) is not None
            })
            sections.append(LoadedSection(
                text=content,
                source=document.get("source")
                or document.get("title", "inline"),
                title=document.get("title", ""),
                section=document.get("section", ""),
                metadata=metadata,
            ))
        return self.add_sections(sections)

    def add_file(
        self,
        filename: str,
        content: bytes,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        suffix = Path(filename).suffix.lower()
        if suffix not in {".txt", ".md", ".markdown", ".pdf"}:
            raise ValueError(f"unsupported document type: {suffix}")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / Path(filename).name
            path.write_bytes(content)
            sections = [
                section.model_copy(update={
                    "source": filename,
                    "metadata": {
                        **section.metadata,
                        **(metadata or {}),
                    },
                })
                for section in load_document(path)
            ]
            return self.add_sections(sections)

    def add_markdown_file(
        self,
        filename: str,
        content: bytes,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        suffix = Path(filename).suffix.lower()
        if suffix not in {".md", ".markdown"}:
            raise ValueError("knowledge upload only supports Markdown files")
        chunks = chunks_from_annotated_markdown(
            filename,
            content,
            chunking_config=self._chunking_config,
            metadata_overrides=metadata,
        )
        return self.add_chunks(chunks)

    def add_sections(self, sections: Iterable[LoadedSection]) -> int:
        loaded = [
            section.model_copy(update={
                "metadata": normalize_knowledge_metadata(
                    section.metadata,
                    source=section.source,
                    title=section.title,
                ),
            })
            for section in sections
        ]
        chunks = chunk_sections(
            loaded,
            chunk_size=self._chunking_config.chunk_size,
            chunk_overlap=self._chunking_config.chunk_overlap,
        )
        if not chunks:
            return 0

        add_chunks = getattr(self.knowledge_base, "add_chunks", None)
        if add_chunks is not None:
            add_chunks(chunks)
        else:
            self.knowledge_base.add_documents([
                {
                    "title": section.title,
                    "content": section.text,
                    "source": section.source,
                    "section": section.section,
                    "page": section.page,
                    "metadata": section.metadata,
                }
                for section in loaded
            ])
        if self._write_dense_on_ingest:
            self.dense_index.add(chunks)
        self._chunks.update({chunk.chunk_id: chunk for chunk in chunks})
        self.refresh()
        return len(chunks)

    def add_chunks(self, chunks: Sequence[DocumentChunk]) -> int:
        if not chunks:
            return 0
        add_chunks = getattr(self.knowledge_base, "add_chunks", None)
        if add_chunks is not None:
            add_chunks(list(chunks))
            if self._write_dense_on_ingest:
                self.dense_index.add(chunks)
        else:
            self.dense_index.add(chunks)
        self._chunks.update({chunk.chunk_id: chunk for chunk in chunks})
        self.refresh()
        return len(chunks)

    def refresh(self) -> None:
        """Reload canonical chunks after version publication or revocation."""
        all_chunks = getattr(self.knowledge_base, "all_chunks", None)
        chunks = (
            all_chunks()
            if all_chunks is not None
            else list(self._chunks.values())
        )
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.sparse_index.add(list(self._chunks.values()))


def local_models_enabled() -> bool:
    """Legacy switch. Prefer RAG_EMBEDDING_PROVIDER when set."""
    value = os.getenv("RAG_LOCAL_MODELS_ENABLED", "true")
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _embedding_provider() -> Optional[str]:
    """Resolve the embedding/reranker provider from environment.

    Returns ``"api"``, ``"local"``, ``"off"`` or ``None`` (fall back to
    ``enable_local_models`` / ``RAG_LOCAL_MODELS_ENABLED``).
    """
    value = os.getenv("RAG_EMBEDDING_PROVIDER", "").strip().lower()
    if value in {"api", "siliconflow"}:
        return "api"
    if value in {"local", "bge"}:
        return "local"
    if value in {"off", "none", "disabled", "bm25"}:
        return "off"
    return None


def _collection_chunks(collection: Any) -> List[DocumentChunk]:
    data = collection.get(include=["documents", "metadatas"])
    chunks = []
    for index, (chunk_id, content, metadata) in enumerate(zip(
        data.get("ids", []),
        data.get("documents", []),
        data.get("metadatas", []),
    )):
        meta = metadata or {}
        reserved = {
            "source", "title", "section", "page",
            "chunk_index", "parent_id",
        }
        chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            content=content,
            source=str(meta.get("source") or meta.get("title") or "knowledge_base"),
            title=str(meta.get("title", "")),
            section=str(meta.get("section", "")),
            page=int(meta.get("page") or 0) or None,
            chunk_index=int(meta.get("chunk_index", index)),
            parent_id=str(meta.get("parent_id") or f"document-{chunk_id}"),
            metadata={
                key: value
                for key, value in meta.items()
                if key not in reserved
            },
        ))
    return chunks
