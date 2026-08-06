import asyncio
import sys
from contextlib import asynccontextmanager
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import UploadFile

from api import main as api_main
from core.customer_agent_runtime import CustomerTurnResult
from rag.indexes import BM25Index
from rag.runtime import KnowledgeRuntime


class FakeKnowledgeBase:
    def __init__(self):
        self.documents = []
        self.chunks = []

    def add_documents(self, documents):
        self.documents.extend(documents)
        return len(documents)

    def add_chunks(self, chunks):
        self.chunks.extend(chunks)
        return len(chunks)


class FakeDense:
    def __init__(self):
        self.added = []

    def add(self, chunks):
        self.added.extend(chunks)


def test_pdf_ingestion_updates_dense_and_bm25(monkeypatch, tmp_path):
    class FakeReader:
        def __init__(self, path):
            self.pages = [SimpleNamespace(extract_text=lambda: "PDF-ONLY 退款政策")]

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=FakeReader))
    knowledge_base = FakeKnowledgeBase()
    dense = FakeDense()
    sparse = BM25Index()
    runtime = KnowledgeRuntime(
        knowledge_base,
        dense,
        sparse,
        SimpleNamespace(),
    )

    count = runtime.add_file("policy.pdf", b"fake-pdf")

    assert count == 1
    assert dense.added[0].page == 1
    assert sparse.search("PDF-ONLY", 1)[0].chunk.page == 1
    assert knowledge_base.chunks[0].section == "page 1"
    assert knowledge_base.chunks[0].page == 1


def test_runtime_build_reuses_canonical_collection_and_wires_reranker(
    monkeypatch,
):
    calls = {}

    class Collection:
        def get(self, include):
            return {"ids": [], "documents": [], "metadatas": []}

    class Client:
        pass

    class Dense(FakeDense):
        def __init__(self, **kwargs):
            super().__init__()
            calls["dense"] = kwargs

        def search(self, query, top_k):
            return []

    class Reranker:
        def __init__(self, model_name):
            calls["reranker"] = model_name

    monkeypatch.setattr("rag.runtime.ChromaDenseIndex", Dense)
    monkeypatch.setattr("rag.runtime.CrossEncoderReranker", Reranker)
    collection = Collection()
    kb = SimpleNamespace(_collection=collection, _client=Client())

    runtime = KnowledgeRuntime.build(
        kb,
        enable_local_models=True,
        embedding_model="bge-test",
        reranker_model="reranker-test",
    )

    assert calls["dense"] == {"collection": collection}
    assert calls["reranker"] == "reranker-test"
    assert runtime.retriever._reranker is not None


def test_pdf_upload_endpoint_uses_runtime_loader(monkeypatch):
    class Runtime:
        def __init__(self):
            self.received = None

        def add_file(self, filename, content):
            self.received = (filename, content)
            return 2

    runtime = Runtime()
    kb = SimpleNamespace(doc_count=12)
    runtime.knowledge_base = kb
    monkeypatch.setattr(api_main, "_knowledge_runtime", runtime)
    upload = UploadFile(file=BytesIO(b"pdf-bytes"), filename="policy.pdf")

    result = asyncio.run(api_main.upload_knowledge(upload))

    assert runtime.received == ("policy.pdf", b"pdf-bytes")
    assert result["added_chunks"] == 2


def test_chat_passes_ordered_memory_context_to_runtime(monkeypatch):
    class MemoryContext:
        recent_messages = []

        def to_prompt_text(self, skill_prompt="", observations=None):
            del observations
            return (
                f"[Skills]\n{skill_prompt}\n\n[当前业务状态]\nrefund\n\n"
                "[最近对话]\nhello\n\n[相关历史]\nhistory\n\n"
                "[用户画像]\nprofile\n\n[Observations]\n[]"
            )

    class Memory:
        async def get_context(self, user_id, conv_id, query):
            return MemoryContext()

        async def add_message(self, *args):
            pass

        async def record_episodic_event(self, *args, **kwargs):
            pass

    class Runtime:
        context = ""

        async def run(self, **kwargs):
            self.context = kwargs["agent_context"]
            return CustomerTurnResult(
                trace_id="trace",
                response="ok",
                intent="query",
                agent_type="knowledge",
                status="completed",
                escalated=False,
                latency_ms=1,
            )

    class Skills:
        def prompt_for(self, message):
            return "skill"

    runtime = Runtime()
    monkeypatch.setattr(api_main, "_memory", Memory())
    monkeypatch.setattr(api_main, "_customer_runtime", runtime)
    monkeypatch.setattr(api_main, "_skill_manager", Skills())

    asyncio.run(api_main.chat(api_main.ChatRequest(message="question")))

    labels = [
        "[Skills]",
        "[当前业务状态]",
        "[最近对话]",
        "[相关历史]",
        "[用户画像]",
        "[Observations]",
    ]
    positions = [runtime.context.index(label) for label in labels]
    assert positions == sorted(positions)


@pytest.mark.asyncio
async def test_lifespan_releases_resources_when_startup_fails(monkeypatch):
    calls = []

    class Resource:
        def __init__(self, name):
            self.name = name

        async def stop(self):
            calls.append(self.name)

        async def close(self):
            calls.append(self.name)

    @asynccontextmanager
    async def failing_components(app):
        del app
        raise RuntimeError("startup failed")
        yield

    monkeypatch.setattr(api_main, "_monitor", Resource("monitor"))
    monkeypatch.setattr(api_main, "_mcp_client", Resource("mcp"))
    monkeypatch.setattr(api_main, "_memory", Resource("memory"))
    monkeypatch.setattr(api_main, "_runtime_components", failing_components)

    with pytest.raises(RuntimeError, match="startup failed"):
        async with api_main.lifespan(api_main.app):
            pass

    assert calls == ["monitor", "mcp", "memory"]


@pytest.mark.asyncio
async def test_lifespan_releases_resources_after_normal_shutdown(monkeypatch):
    calls = []

    class Resource:
        def __init__(self, name):
            self.name = name

        async def stop(self):
            calls.append(self.name)

        async def close(self):
            calls.append(self.name)

    @asynccontextmanager
    async def components(app):
        del app
        yield

    monkeypatch.setattr(api_main, "_monitor", Resource("monitor"))
    monkeypatch.setattr(api_main, "_mcp_client", Resource("mcp"))
    monkeypatch.setattr(api_main, "_memory", Resource("memory"))
    monkeypatch.setattr(api_main, "_runtime_components", components)

    async with api_main.lifespan(api_main.app):
        pass

    assert calls == ["monitor", "mcp", "memory"]
