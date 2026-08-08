import threading

from mcp.knowledge_base import KnowledgeBase
from rag.indexes import BM25Index
from rag.models import DocumentChunk
from rag.versioning import MAX_TIMESTAMP, RetrievalFilter


class FakeCollection:
    def __init__(self):
        self.records = {}

    def count(self):
        return len(self.records)

    def upsert(self, ids, documents, metadatas):
        for chunk_id, document, metadata in zip(ids, documents, metadatas):
            self.records[chunk_id] = {
                "document": document,
                "metadata": dict(metadata),
            }

    def update(self, ids, metadatas):
        for chunk_id, metadata in zip(ids, metadatas):
            self.records[chunk_id]["metadata"] = dict(metadata)

    def delete(self, ids):
        for chunk_id in ids:
            self.records.pop(chunk_id, None)

    def get(self, where=None, include=None):
        selected = [
            (chunk_id, value)
            for chunk_id, value in self.records.items()
            if _matches(value["metadata"], where)
        ]
        return {
            "ids": [item[0] for item in selected],
            "documents": [item[1]["document"] for item in selected],
            "metadatas": [item[1]["metadata"] for item in selected],
        }


def _matches(metadata, where):
    if not where:
        return True
    if "$and" in where:
        return all(_matches(metadata, condition) for condition in where["$and"])
    key, expected = next(iter(where.items()))
    if isinstance(expected, dict):
        if "$eq" in expected:
            return metadata.get(key) == expected["$eq"]
        if "$in" in expected:
            return metadata.get(key) in expected["$in"]
    return metadata.get(key) == expected


def make_chunk(chunk_id, version, effective_at, status="published"):
    return DocumentChunk(
        chunk_id=chunk_id,
        content=f"退款政策 {version}",
        source="refund.md",
        title="退款政策",
        section="退款",
        chunk_index=0,
        parent_id=f"parent-{chunk_id}",
        metadata={
            "knowledge_id": "refund-policy",
            "version": version,
            "version_id": f"refund-policy:{version}",
            "status": status,
            "effective_at": effective_at,
            "expires_at": MAX_TIMESTAMP,
        },
    )


def make_knowledge_base():
    knowledge_base = KnowledgeBase.__new__(KnowledgeBase)
    knowledge_base._collection = FakeCollection()
    knowledge_base._version_lock = threading.RLock()
    return knowledge_base


def test_version_timeline_supports_current_and_historical_retrieval():
    knowledge_base = make_knowledge_base()
    knowledge_base.add_chunks([make_chunk("v1", "v1", 100.0)])
    knowledge_base.add_chunks([make_chunk("v2", "v2", 200.0)])

    versions = knowledge_base.list_versions("refund-policy")
    chunks = knowledge_base.all_chunks()
    index = BM25Index()
    index.add(chunks)

    assert versions[0]["version"] == "v2"
    assert versions[0]["is_current"] is True
    assert versions[1]["expires_at"] == 200.0
    assert [
        hit.chunk.metadata["version"]
        for hit in index.search(
            "退款政策",
            5,
            filters=RetrievalFilter.current(as_of=150.0),
        )
    ] == ["v1"]
    assert [
        hit.chunk.metadata["version"]
        for hit in index.search(
            "退款政策",
            5,
            filters=RetrievalFilter.current(as_of=250.0),
        )
    ] == ["v2"]


def test_revoke_restores_previous_published_version():
    knowledge_base = make_knowledge_base()
    knowledge_base.add_chunks([make_chunk("v1", "v1", 100.0)])
    knowledge_base.add_chunks([make_chunk("v2", "v2", 200.0)])

    revoked = knowledge_base.revoke_version("refund-policy", "v2")
    versions = knowledge_base.list_versions("refund-policy")

    assert revoked["status"] == "revoked"
    assert versions[1]["version"] == "v1"
    assert versions[1]["is_current"] is True
    assert versions[1]["expires_at"] == MAX_TIMESTAMP


def test_publishing_draft_version_rebuilds_timeline():
    knowledge_base = make_knowledge_base()
    knowledge_base.add_chunks([make_chunk("v1", "v1", 100.0)])
    knowledge_base.add_chunks([
        make_chunk("v2", "v2", 200.0, status="draft"),
    ])

    before = knowledge_base.list_versions("refund-policy")
    published = knowledge_base.publish_version(
        "refund-policy",
        "v2",
        effective_at=250.0,
    )
    after = knowledge_base.list_versions("refund-policy")

    assert before[0]["status"] == "draft"
    assert published["status"] == "published"
    assert after[1]["expires_at"] == 250.0


def test_reingesting_same_version_preserves_lifecycle_metadata():
    knowledge_base = make_knowledge_base()
    knowledge_base.add_chunks([make_chunk("v1-old", "v1", 100.0)])
    knowledge_base.add_chunks([make_chunk("v1-new", "v1", 999.0)])

    versions = knowledge_base.list_versions("refund-policy")
    chunks = knowledge_base.all_chunks()

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "v1-new"
    assert versions[0]["effective_at"] == 100.0
