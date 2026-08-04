import sys
from types import SimpleNamespace

import pytest

from core.tool_registry import ToolRegistry, ToolType
from rag.indexes import BM25Index, ChromaDenseIndex
from rag.loaders import chunk_sections, load_document
from rag.models import DocumentChunk, SearchHit
from rag.retriever import HybridRetriever, reciprocal_rank_fusion
from rag.tool import register_rag_tool


def make_chunk(
    chunk_id: str,
    content: str,
    *,
    source: str = "policy.md",
    section: str = "退款",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        content=content,
        source=source,
        title="客服政策",
        section=section,
        chunk_index=0,
        parent_id=f"parent-{chunk_id}",
    )


class FakeDenseIndex:
    def __init__(self, hits=None):
        self.chunks = []
        self.hits = hits or []

    def add(self, chunks):
        self.chunks = list(chunks)

    def search(self, query, top_k):
        return self.hits[:top_k]


class FakeReranker:
    def score(self, query, chunks):
        return [0.2 if chunk.chunk_id == "a" else 0.9 for chunk in chunks]


class FakeCollection:
    def __init__(self):
        self.upsert_payload = None

    def upsert(self, **payload):
        self.upsert_payload = payload

    def query(self, **params):
        return {
            "ids": [["a"]],
            "documents": [["7 天内可退款"]],
            "metadatas": [[{
                "source": "policy.md",
                "title": "退款政策",
                "section": "期限",
                "page": 0,
                "chunk_index": 2,
                "parent_id": "parent-a",
                "version": "v1",
            }]],
            "distances": [[0.1]],
        }


def test_txt_loader_and_chunk_metadata(tmp_path):
    path = tmp_path / "guide.txt"
    path.write_text("退款说明。" * 30, encoding="utf-8")

    chunks = chunk_sections(load_document(path), chunk_size=40, chunk_overlap=5)

    assert len(chunks) > 1
    assert all(chunk.source == str(path) for chunk in chunks)
    assert all(chunk.title == "guide" for chunk in chunks)
    assert all(chunk.parent_id for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_markdown_loader_preserves_heading_path(tmp_path):
    path = tmp_path / "policy.md"
    path.write_text(
        "# 售后政策\n\n总则。\n\n## 退款\n\n七天可退。\n\n### 运费\n\n质量问题免运费。",
        encoding="utf-8",
    )

    sections = load_document(path)

    assert sections[0].title == "售后政策"
    assert sections[1].section == "售后政策 > 退款"
    assert sections[2].section == "售后政策 > 退款 > 运费"


def test_pdf_loader_preserves_page(monkeypatch, tmp_path):
    class FakeReader:
        def __init__(self, path):
            self.pages = [
                SimpleNamespace(extract_text=lambda: "第一页"),
                SimpleNamespace(extract_text=lambda: ""),
            ]

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=FakeReader))
    path = tmp_path / "manual.pdf"
    path.write_bytes(b"fake")

    sections = load_document(path)

    assert len(sections) == 1
    assert sections[0].page == 1
    assert sections[0].section == "page 1"


def test_loader_rejects_unsupported_type(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported"):
        load_document(path)


def test_chunk_configuration_is_validated():
    with pytest.raises(ValueError):
        chunk_sections([], chunk_size=10, chunk_overlap=10)


def test_chroma_dense_index_uses_injected_collection():
    collection = FakeCollection()
    index = ChromaDenseIndex(collection=collection)
    chunk = make_chunk("a", "7 天内可退款")

    index.add([chunk])
    hits = index.search("退款期限", 3)

    assert collection.upsert_payload["ids"] == ["a"]
    assert hits[0].dense_score == pytest.approx(0.9)
    assert hits[0].chunk.metadata == {"version": "v1"}


def test_chroma_dense_index_requires_client_or_collection():
    with pytest.raises(ValueError):
        ChromaDenseIndex()


def test_bm25_exact_identifier_retrieval():
    index = BM25Index()
    exact = make_chunk("exact", "订单 ERROR-401 登录失败")
    generic = make_chunk("generic", "常见登录问题处理方式")
    index.add([generic, exact])

    hits = index.search("ERROR-401", 2)

    assert hits[0].chunk.chunk_id == "exact"
    assert hits[0].bm25_score > 0


def test_bm25_replaces_existing_documents_on_add():
    index = BM25Index()
    index.add([make_chunk("old", "LEGACY-ONLY 旧政策")])
    index.add([make_chunk("new", "CURRENT-ONLY 新政策")])

    assert index.search("LEGACY-ONLY", 5) == []
    assert index.search("CURRENT-ONLY", 5)[0].chunk.chunk_id == "new"


def test_rrf_merges_scores_and_keeps_channel_scores():
    a = make_chunk("a", "退款")
    b = make_chunk("b", "退货")
    dense = [
        SearchHit(chunk=a, score=0.9, dense_score=0.9),
        SearchHit(chunk=b, score=0.8, dense_score=0.8),
    ]
    sparse = [
        SearchHit(chunk=b, score=2.0, bm25_score=2.0),
        SearchHit(chunk=a, score=1.0, bm25_score=1.0),
    ]

    fused = reciprocal_rank_fusion(dense, sparse)

    assert {hit.chunk.chunk_id for hit in fused} == {"a", "b"}
    assert all(hit.rrf_score for hit in fused)
    assert all(hit.dense_score is not None and hit.bm25_score is not None for hit in fused)


def test_hybrid_retriever_reranks_and_builds_citations():
    a = make_chunk("a", "一般说明")
    b = make_chunk("b", "退款期限为七天", section="退款 > 期限")
    dense = FakeDenseIndex([
        SearchHit(chunk=a, score=0.9, dense_score=0.9),
        SearchHit(chunk=b, score=0.8, dense_score=0.8),
    ])
    sparse = BM25Index()
    sparse.add([a, b])
    retriever = HybridRetriever(dense, sparse, reranker=FakeReranker())

    result = retriever.search("退款期限", top_k=1)

    assert result.answered is True
    assert result.hits[0].chunk.chunk_id == "b"
    assert result.hits[0].rerank_score == 0.9
    assert result.citations[0].citation_id == "[1]"
    assert result.citations[0].section == "退款 > 期限"


def test_dense_only_mode_does_not_call_sparse():
    class FailingSparse:
        def add(self, chunks):
            pass

        def search(self, query, top_k):
            raise AssertionError("sparse index must not be called")

    chunk = make_chunk("a", "退款说明")
    dense = FakeDenseIndex([SearchHit(chunk=chunk, score=0.7, dense_score=0.7)])
    retriever = HybridRetriever(dense, FailingSparse())

    result = retriever.search("退款", use_sparse=False, use_reranker=False)

    assert result.answered is True


def test_no_answer_when_below_threshold():
    chunk = make_chunk("a", "无关信息")
    dense = FakeDenseIndex([SearchHit(chunk=chunk, score=0.1, dense_score=0.1)])
    retriever = HybridRetriever(dense, BM25Index(), relevance_threshold=0.5)

    result = retriever.search(
        "退款政策", use_sparse=False, use_reranker=False
    )

    assert result.answered is False
    assert result.hits == []
    assert result.reason == "no_relevant_evidence"


@pytest.mark.asyncio
async def test_rag_search_is_registered_as_read_only_tool():
    chunk = make_chunk("a", "七天内可退款")
    dense = FakeDenseIndex([SearchHit(chunk=chunk, score=0.9, dense_score=0.9)])
    retriever = HybridRetriever(dense, BM25Index())
    registry = ToolRegistry()

    register_rag_tool(registry, retriever, agent_names=["knowledge"])
    result = await registry.call(
        "knowledge",
        "rag_search",
        {"query": "退款", "mode": "dense", "top_k": 1},
    )

    assert registry.get_spec("rag_search").tool_type == ToolType.READ
    assert result.success is True
    assert result.data["answered"] is True
    assert result.data["citations"][0]["source"] == "policy.md"


@pytest.mark.asyncio
async def test_rag_tool_obeys_agent_whitelist():
    retriever = HybridRetriever(FakeDenseIndex(), BM25Index())
    registry = ToolRegistry()
    register_rag_tool(registry, retriever, agent_names=["knowledge"])

    result = await registry.call("order", "rag_search", {"query": "退款"})

    assert result.success is False
    assert "not allowed" in result.error
