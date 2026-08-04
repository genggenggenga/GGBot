"""Tests for Task 8: memory data source boundaries, overwrite summarization,
event-triggered episodic memory, stable preference gating.

No real Redis, ChromaDB, LLM, or network required.
"""
import asyncio
import json
from typing import Any, Dict, List, Optional

import pytest

from core.agent_models import DialogueState
from core.state_store import InMemoryStateStore
from memory.conversation_memory import (
    EpisodicEventType,
    MemoryContext,
    MemoryManager,
    MsgRole,
    SUMMARY_MAX_CHARS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fakes for external dependencies
# ═══════════════════════════════════════════════════════════════════════════════

class FakeRedis:
    """Minimal fake Redis supporting lists, strings, and expiry."""

    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.ttls: Dict[str, int] = {}

    # -- string --
    def get(self, key: str) -> Optional[str]:
        return self.data.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.data[key] = value
        self.ttls[key] = ttl

    def delete(self, *keys: str) -> None:
        for k in keys:
            self.data.pop(k, None)
            self.ttls.pop(k, None)

    # -- list --
    def lpush(self, key: str, *values: str) -> int:
        if key not in self.data or not isinstance(self.data[key], list):
            self.data[key] = []
        for v in values:
            self.data[key].insert(0, v)
        return len(self.data[key])

    def lrange(self, key: str, start: int, end: int) -> List[str]:
        lst = self.data.get(key, [])
        if not isinstance(lst, list):
            return []
        if end < 0:
            end = len(lst) + end + 1
        else:
            end = end + 1
        return lst[start:end]

    def llen(self, key: str) -> int:
        lst = self.data.get(key, [])
        return len(lst) if isinstance(lst, list) else 0

    def expire(self, key: str, seconds: int) -> bool:
        self.ttls[key] = seconds
        return True


class _FakeCollection:
    """Fake ChromaDB collection."""

    def __init__(self, name: str):
        self.name = name
        self._ids: List[str] = []
        self._docs: List[str] = []
        self._metas: List[Dict[str, Any]] = []

    def add(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]] = None):
        for i, doc_id in enumerate(ids):
            if doc_id in self._ids:
                idx = self._ids.index(doc_id)
                self._docs[idx] = documents[i]
                if metadatas:
                    self._metas[idx] = metadatas[i]
            else:
                self._ids.append(doc_id)
                self._docs.append(documents[i])
                if metadatas:
                    self._metas.append(metadatas[i])
                else:
                    self._metas.append({})

    def delete(self, ids: List[str] = None):
        if not ids:
            return
        for doc_id in ids:
            if doc_id in self._ids:
                idx = self._ids.index(doc_id)
                self._ids.pop(idx)
                self._docs.pop(idx)
                self._metas.pop(idx)

    def get(self, where: Dict[str, Any] = None, limit: int = 1):
        if not self._docs:
            return {"ids": [], "documents": [], "metadatas": []}
        # Simple where filter: match exact user_id
        if where and "user_id" in where:
            uid = where["user_id"]
            filtered = [
                (i, doc, meta)
                for i, (doc, meta) in enumerate(zip(self._docs, self._metas))
                if meta.get("user_id") == uid
            ]
            if not filtered:
                return {"ids": [], "documents": [], "metadatas": []}
            # Return latest
            i, doc, meta = filtered[-1]
            return {
                "ids": [self._ids[i]],
                "documents": [doc],
                "metadatas": [meta],
            }
        # Return latest
        return {
            "ids": [self._ids[-1]],
            "documents": [self._docs[-1]],
            "metadatas": [self._metas[-1]],
        }

    def query(self, query_texts: List[str], n_results: int = 5, where: Dict[str, Any] = None):
        if not self._docs:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]]}
        # Simple dummy: return all docs matching where (no real vector search)
        results = []
        if where and "user_id" in where:
            uid = where["user_id"]
            for i, (doc, meta) in enumerate(zip(self._docs, self._metas)):
                if meta.get("user_id") == uid:
                    results.append(doc)
        else:
            results = list(self._docs)
        results = results[:n_results]
        return {"ids": [results], "documents": [results], "metadatas": [[] for _ in results]}


class FakeChroma:
    """Minimal fake ChromaDB client."""

    def __init__(self):
        self._collections: Dict[str, _FakeCollection] = {}

    def get_or_create_collection(self, name: str, **kwargs) -> _FakeCollection:
        if name not in self._collections:
            self._collections[name] = _FakeCollection(name)
        return self._collections[name]

    def heartbeat(self):
        return True


class FakeLLMClient:
    """Fake AsyncAnthropic client that returns controllable responses."""

    def __init__(self, summary_response: str = "测试摘要内容", profile_response: str = None):
        self.summary_response = summary_response
        self.profile_response = profile_response or json.dumps({
            "preferences": [],
            "language": "zh",
        })
        self.calls: List[Dict[str, Any]] = []
        self.messages = self  # mimic .messages attribute

    async def create(self, model: str, max_tokens: int, temperature: float, messages: List[Dict]):
        self.calls.append({"model": model, "messages": messages})
        # Determine response type by prompt content
        prompt_text = ""
        for m in messages:
            if isinstance(m.get("content"), str):
                prompt_text += m["content"]
        if "稳定偏好" in prompt_text or "preferences" in prompt_text:
            content_text = self.profile_response
        else:
            content_text = self.summary_response

        class FakeContent:
            def __init__(self, text):
                self.text = text
            def __getitem__(self, idx):
                return self

        class FakeResponse:
            def __init__(self, text):
                self.content = [type("Block", (), {"text": text})()]
        return FakeResponse(content_text)


def _make_manager(
    summary_response: str = "新的覆盖摘要",
    profile_response: str = None,
    enable_state_store: bool = False,
) -> MemoryManager:
    """Build a MemoryManager wired with fakes."""
    fake_redis = FakeRedis()
    fake_chroma = FakeChroma()
    fake_llm = FakeLLMClient(summary_response=summary_response, profile_response=profile_response)

    state_store = InMemoryStateStore() if enable_state_store else None

    mgr = MemoryManager(
        api_key="fake-key",
        redis_client=fake_redis,
        chroma_client=fake_chroma,
        state_store=state_store,
    )
    # Replace LLM client with fake
    mgr._client = fake_llm
    return mgr


# ═══════════════════════════════════════════════════════════════════════════════
# 8.1 Data source boundaries
# ═══════════════════════════════════════════════════════════════════════════════

class TestMemoryContextDataSources:
    def test_four_distinct_sources_in_context(self):
        """MemoryContext should separate recent_messages, relevant_history,
        user_profile, summary, and dialogue_state as distinct sources."""
        ctx = MemoryContext(
            recent_messages=[],
            relevant_history=["历史1"],
            user_profile={"language": "zh"},
            summary="会话摘要",
            dialogue_state=DialogueState(active_intent="refund_request"),
        )
        assert ctx.relevant_history == ["历史1"]
        assert ctx.user_profile == {"language": "zh"}
        assert ctx.summary == "会话摘要"
        assert ctx.dialogue_state is not None
        assert ctx.dialogue_state.active_intent == "refund_request"

    def test_dialogue_state_optional_for_backwards_compat(self):
        """Old code that does not pass dialogue_state should still work."""
        ctx = MemoryContext(
            recent_messages=[],
            relevant_history=[],
            user_profile={},
            summary="",
        )
        assert ctx.dialogue_state is None
        text = ctx.to_prompt_text()
        assert "当前业务状态" not in text

    def test_to_prompt_text_includes_dialogue_state_when_present(self):
        ctx = MemoryContext(
            recent_messages=[],
            relevant_history=[],
            user_profile={},
            summary="",
            dialogue_state=DialogueState(active_intent="refund_request", slots={"order_id": "ORD-1001"}),
        )
        text = ctx.to_prompt_text()
        assert "当前业务状态" in text
        assert "refund_request" in text
        assert "ORD-1001" in text

    def test_to_prompt_text_empty_state_not_included(self):
        """Empty DialogueState with no fields should not produce an empty section."""
        ctx = MemoryContext(
            recent_messages=[],
            relevant_history=[],
            user_profile={},
            summary="",
            dialogue_state=DialogueState(),
        )
        text = ctx.to_prompt_text()
        # Empty state has no non-none fields, so no section
        assert "当前业务状态" not in text

    @pytest.mark.asyncio
    async def test_get_context_loads_dialogue_state_from_store_when_configured(self):
        mgr = _make_manager(enable_state_store=True)
        state = DialogueState(active_intent="order_query", slots={"order_id": "ORD-999"})
        await mgr._state_store.save("u1", "c1", state)

        ctx = await mgr.get_context("u1", "c1", query="查订单")
        assert ctx.dialogue_state is not None
        assert ctx.dialogue_state.active_intent == "order_query"
        assert ctx.dialogue_state.slots["order_id"] == "ORD-999"

    @pytest.mark.asyncio
    async def test_get_context_explicit_state_overrides_store(self):
        mgr = _make_manager(enable_state_store=True)
        stored_state = DialogueState(active_intent="order_query")
        await mgr._state_store.save("u1", "c1", stored_state)

        explicit_state = DialogueState(active_intent="refund_request")
        ctx = await mgr.get_context("u1", "c1", query="退款", dialogue_state=explicit_state)
        assert ctx.dialogue_state.active_intent == "refund_request"

    @pytest.mark.asyncio
    async def test_get_context_no_state_store_does_not_fail(self):
        mgr = _make_manager(enable_state_store=False)
        ctx = await mgr.get_context("u1", "c1")
        assert ctx.dialogue_state is None
        assert ctx.recent_messages == []
        assert ctx.user_profile == {}


# ═══════════════════════════════════════════════════════════════════════════════
# 8.2 Overwrite-style summarization (bounded, not appending)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOverwriteSummarization:
    @pytest.mark.asyncio
    async def test_summary_overwrites_not_appends(self):
        """After compression, the stored summary must be exactly the new summary,
        not a concatenation of old and new."""
        mgr = _make_manager(summary_response="全新摘要内容")

        # Seed an "old summary"
        skey = mgr._summary_key("u1", "c1")
        mgr._redis.setex(skey, 86400, "旧的摘要内容")

        # Add enough messages to trigger compression (COMPRESS_AT = 15)
        for i in range(mgr.COMPRESS_AT):
            await mgr.add_message("u1", "c1", MsgRole.USER, f"消息{i}")

        new_summary = mgr._redis.get(skey)
        assert new_summary == "全新摘要内容"
        assert "旧的摘要内容" not in new_summary

    @pytest.mark.asyncio
    async def test_summary_is_bounded_by_max_chars(self):
        """Even if LLM returns a long response, summary must be truncated to SUMMARY_MAX_CHARS."""
        long_summary = "A" * (SUMMARY_MAX_CHARS + 200)
        mgr = _make_manager(summary_response=long_summary)

        for i in range(mgr.COMPRESS_AT):
            await mgr.add_message("u1", "c1", MsgRole.USER, f"消息{i}")

        skey = mgr._summary_key("u1", "c1")
        stored = mgr._redis.get(skey)
        assert len(stored) <= SUMMARY_MAX_CHARS + 1  # +1 for possible ellipsis "…"
        assert stored.endswith("…") or len(stored) == SUMMARY_MAX_CHARS

    @pytest.mark.asyncio
    async def test_compress_keeps_only_recent_messages(self):
        """After compression, working memory should retain exactly KEEP_RECENT messages."""
        mgr = _make_manager()

        for i in range(mgr.COMPRESS_AT):
            await mgr.add_message("u1", "c1", MsgRole.USER, f"消息{i}")

        wm_key = mgr._wm_key("u1", "c1")
        wm_len = mgr._redis.llen(wm_key)
        assert wm_len == mgr.KEEP_RECENT

    @pytest.mark.asyncio
    async def test_compress_does_not_write_episodic_memory(self):
        """Compression alone must NOT write episodic memory
        (episodic memory is event-triggered only)."""
        mgr = _make_manager()

        for i in range(mgr.COMPRESS_AT):
            await mgr.add_message("u1", "c1", MsgRole.USER, f"消息{i}")

        episodic = mgr._episodic
        # Should not contain any entries (compression does not write episodic)
        assert len(episodic._docs) == 0

    @pytest.mark.asyncio
    async def test_multiple_compressions_stay_bounded(self):
        """After multiple compression cycles, summary does not grow unbounded."""
        mgr = _make_manager(summary_response="短摘要")

        # First compression
        for i in range(mgr.COMPRESS_AT):
            await mgr.add_message("u1", "c1", MsgRole.USER, f"第1轮{i}")
        skey = mgr._summary_key("u1", "c1")
        after_first = mgr._redis.get(skey)

        # Add more messages to trigger second compression
        for i in range(mgr.COMPRESS_AT - mgr.KEEP_RECENT):
            await mgr.add_message("u1", "c1", MsgRole.USER, f"第2轮{i}")
        after_second = mgr._redis.get(skey)

        # Summary is overwritten, so second compression result is just the new summary
        # (LLM may have seen old summary as context, but stored value is the fresh one)
        assert after_second == "短摘要"
        assert len(after_second) <= SUMMARY_MAX_CHARS


# ═══════════════════════════════════════════════════════════════════════════════
# 8.3 Event-triggered episodic memory (task_completed / handoff)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEventTriggeredEpisodicMemory:
    @pytest.mark.asyncio
    async def test_record_task_completed_writes_episodic(self):
        mgr = _make_manager(summary_response="退款已处理完成")
        await mgr.add_message("u1", "c1", MsgRole.USER, "我要退款，订单号ORD-1001")
        await mgr.add_message("u1", "c1", MsgRole.ASSISTANT, "退款已为您申请成功")

        await mgr.record_episodic_event(
            "u1", "c1",
            event_type=EpisodicEventType.TASK_COMPLETED,
        )

        episodic = mgr._episodic
        assert len(episodic._docs) == 1
        doc = episodic._docs[0]
        assert "任务完成" in doc
        meta = episodic._metas[0]
        assert meta["event_type"] == "task_completed"
        assert meta["user_id"] == "u1"
        assert meta["conv_id"] == "c1"

    @pytest.mark.asyncio
    async def test_record_handoff_writes_episodic(self):
        mgr = _make_manager(summary_response="需要人工处理退款问题")
        await mgr.add_message("u1", "c1", MsgRole.USER, "你们系统有问题")

        await mgr.record_episodic_event(
            "u1", "c1",
            event_type=EpisodicEventType.HANDOFF,
            summary="用户反馈系统异常，已转人工",
        )

        episodic = mgr._episodic
        assert len(episodic._docs) == 1
        doc = episodic._docs[0]
        assert "转人工" in doc
        assert "用户反馈系统异常" in doc
        meta = episodic._metas[0]
        assert meta["event_type"] == "handoff"

    @pytest.mark.asyncio
    async def test_compress_does_not_trigger_episodic(self):
        """Compression must not write episodic; only explicit event calls do."""
        mgr = _make_manager()
        for i in range(mgr.COMPRESS_AT):
            await mgr.add_message("u1", "c1", MsgRole.USER, f"msg{i}")

        assert len(mgr._episodic._docs) == 0

        # Now explicitly record an event
        await mgr.record_episodic_event(
            "u1", "c1", EpisodicEventType.TASK_COMPLETED, summary="done"
        )
        assert len(mgr._episodic._docs) == 1

    @pytest.mark.asyncio
    async def test_record_episodic_with_provided_summary_skips_llm(self):
        """If summary is provided, record_episodic_event should not call LLM."""
        mgr = _make_manager()
        await mgr.add_message("u1", "c1", MsgRole.USER, "测试")

        await mgr.record_episodic_event(
            "u1", "c1",
            event_type=EpisodicEventType.TASK_COMPLETED,
            summary="手动提供的摘要",
        )
        # LLM should not have been called
        assert len(mgr._client.calls) == 0
        doc = mgr._episodic._docs[0]
        assert "手动提供的摘要" in doc

    @pytest.mark.asyncio
    async def test_record_episodic_with_metadata(self):
        mgr = _make_manager(summary_response="摘要")
        await mgr.add_message("u1", "c1", MsgRole.USER, "测试")
        await mgr.record_episodic_event(
            "u1", "c1",
            event_type=EpisodicEventType.HANDOFF,
            metadata={"reason": "max_steps_exceeded", "intent": "refund_request"},
        )
        meta = mgr._episodic._metas[0]
        assert meta["reason"] == "max_steps_exceeded"
        assert meta["intent"] == "refund_request"

    @pytest.mark.asyncio
    async def test_episodic_search_filters_by_user_id(self):
        """Episodic search should only return results for the requesting user."""
        mgr = _make_manager(summary_response="摘要")
        await mgr.add_message("u1", "c1", MsgRole.USER, "用户1的问题")
        await mgr.record_episodic_event("u1", "c1", EpisodicEventType.TASK_COMPLETED, summary="用户1完成")
        await mgr.add_message("u2", "c2", MsgRole.USER, "用户2的问题")
        await mgr.record_episodic_event("u2", "c2", EpisodicEventType.TASK_COMPLETED, summary="用户2完成")

        results = await mgr._search_episodic("u1", "问题")
        assert len(results) == 1
        assert "用户1完成" in results[0]


# ═══════════════════════════════════════════════════════════════════════════════
# 8.4 Stable preference gating
# ═══════════════════════════════════════════════════════════════════════════════

class TestStablePreferenceGating:
    @pytest.mark.asyncio
    async def test_no_preference_signal_skips_profile_update(self):
        """If recent messages contain no stable preference keyword, LLM is not called."""
        mgr = _make_manager()
        # Messages about a specific order (transient) but no preference signal
        await mgr.add_message("u1", "c1", MsgRole.USER, "我要退款，订单号ORD-1001")
        await mgr.add_message("u1", "c1", MsgRole.ASSISTANT, "好的帮您查")

        await mgr.update_profile("u1", "c1")
        # LLM should NOT have been called
        assert len(mgr._client.calls) == 0
        # Profile should be empty
        profile = await mgr._get_profile("u1")
        assert profile == {}

    @pytest.mark.asyncio
    async def test_stable_preference_keyword_triggers_llm(self):
        """Messages containing stable preference keywords trigger LLM call."""
        profile_resp = json.dumps({
            "preferences": ["用中文回复"],
            "language": "zh",
        })
        mgr = _make_manager(profile_response=profile_resp)
        await mgr.add_message("u1", "c1", MsgRole.USER, "以后都请用中文回复我")

        await mgr.update_profile("u1", "c1")
        # LLM should have been called
        assert len(mgr._client.calls) >= 1
        profile = await mgr._get_profile("u1")
        assert "preferences" in profile
        assert "language" in profile

    @pytest.mark.asyncio
    async def test_transient_entities_filtered_from_profile(self):
        """Even if LLM returns transient data (order_id, status), the gate filters it out."""
        profile_resp = json.dumps({
            "preferences": ["快速处理订单ORD-1001", "用中文"],
            "language": "zh",
            "order_id": "ORD-1001",  # non-whitelisted field
            "entities": {"产品": ["手机"]},  # non-whitelisted
        })
        mgr = _make_manager(profile_response=profile_resp)
        await mgr.add_message("u1", "c1", MsgRole.USER, "我希望请用中文回复，订单ORD-1001退款")

        await mgr.update_profile("u1", "c1")
        profile = await mgr._get_profile("u1")
        # order_id and entities should NOT be in profile (not whitelisted)
        assert "order_id" not in profile
        assert "entities" not in profile
        # preference containing order number should be filtered out
        prefs = profile.get("preferences", [])
        assert all("ORD-1001" not in p for p in prefs)
        # language should be preserved
        assert profile.get("language") == "zh"

    @pytest.mark.asyncio
    async def test_profile_merges_with_existing_preferences(self):
        """Multiple update_profile calls should merge preferences without duplicates."""
        profile_resp1 = json.dumps({
            "preferences": ["用中文回复"],
            "language": "zh",
        })
        profile_resp2 = json.dumps({
            "preferences": ["发短信联系"],
            "communication_preference": "sms",
        })
        mgr = _make_manager(profile_response=profile_resp1)
        await mgr.add_message("u1", "c1", MsgRole.USER, "以后请用中文")
        await mgr.update_profile("u1", "c1")

        # Switch fake response for second call
        mgr._client.profile_response = profile_resp2
        await mgr.add_message("u1", "c1", MsgRole.USER, "以后请发短信联系我")
        await mgr.update_profile("u1", "c1")

        profile = await mgr._get_profile("u1")
        assert profile["language"] == "zh"
        assert profile["communication_preference"] == "sms"
        prefs = profile["preferences"]
        assert "用中文回复" in prefs
        assert "发短信联系" in prefs
        # No duplicates
        assert len(prefs) == len(set(prefs))

    def test_contains_transient_detects_order_id(self):
        mgr = _make_manager()
        assert mgr._contains_transient("订单号是ORD-1001234") is True
        assert mgr._contains_transient("你好，今天天气怎么样") is False

    def test_contains_transient_detects_status(self):
        mgr = _make_manager()
        assert mgr._contains_transient("我的订单已发货") is True
        assert mgr._contains_transient("请用英文回复") is False

    def test_contains_transient_detects_logistics_terms(self):
        mgr = _make_manager()
        assert mgr._contains_transient("帮我查物流SF1234567890") is True

    def test_filter_profile_data_removes_non_whitelisted_fields(self):
        mgr = _make_manager()
        raw = {
            "preferences": ["用中文"],
            "language": "zh",
            "entities": {"产品": ["手机"]},  # not whitelisted
            "random_field": "x",
        }
        filtered = mgr._filter_profile_data(raw)
        assert "preferences" in filtered
        assert "language" in filtered
        assert "entities" not in filtered
        assert "random_field" not in filtered


# ═══════════════════════════════════════════════════════════════════════════════
# 8.5 Context ordering and API compatibility
# ═══════════════════════════════════════════════════════════════════════════════

class TestContextOrdering:
    @pytest.mark.asyncio
    async def test_to_prompt_text_ordering(self):
        """Context order: Skill → state → recent messages → memory → observations."""
        from memory.conversation_memory import Message
        from core.agent_models import Observation
        ctx = MemoryContext(
            recent_messages=[Message(role=MsgRole.USER, content="最近用户消息")],
            relevant_history=["历史片段1"],
            user_profile={"language": "zh"},
            summary="会话摘要内容",
            dialogue_state=DialogueState(active_intent="refund_request"),
        )
        text = ctx.to_prompt_text(
            skill_prompt="退款处理规范",
            observations=[Observation(
                source="tool",
                name="query_order",
                success=True,
                data={"status": "delivered"},
            )],
        )
        idx_skill = text.find("[Skills]")
        idx_summary = text.find("[会话摘要]")
        idx_state = text.find("[当前业务状态]")
        idx_history = text.find("[相关历史]")
        idx_profile = text.find("[用户画像]")
        idx_recent = text.find("[最近对话]")
        idx_observations = text.find("[Observations]")
        assert (
            idx_skill
            < idx_state
            < idx_recent
            < idx_summary
            < idx_history
            < idx_profile
            < idx_observations
        )

    @pytest.mark.asyncio
    async def test_api_backwards_compatible_no_chrome_redis_required_signature(self):
        """Old code constructing MemoryManager with the original parameters should still work."""
        # This should not raise; we use fakes internally for testing
        mgr = _make_manager()
        assert mgr is not None
        # Old API methods exist
        assert hasattr(mgr, "add_message")
        assert hasattr(mgr, "get_context")
        assert hasattr(mgr, "update_profile")
        assert callable(mgr.add_message)
        assert callable(mgr.get_context)
        assert callable(mgr.update_profile)

    @pytest.mark.asyncio
    async def test_get_context_returns_correct_recent_messages(self):
        mgr = _make_manager()
        await mgr.add_message("u1", "c1", MsgRole.USER, "你好")
        await mgr.add_message("u1", "c1", MsgRole.ASSISTANT, "您好，请问有什么可以帮您")

        ctx = await mgr.get_context("u1", "c1")
        assert len(ctx.recent_messages) == 2
        assert ctx.recent_messages[0].role == MsgRole.USER
        assert ctx.recent_messages[0].content == "你好"
        assert ctx.recent_messages[1].role == MsgRole.ASSISTANT

    @pytest.mark.asyncio
    async def test_memory_context_to_prompt_text_old_format_still_works(self):
        """When dialogue_state is None (old behavior), output matches legacy format."""
        from memory.conversation_memory import Message
        ctx = MemoryContext(
            recent_messages=[Message(role=MsgRole.USER, content="你好")],
            relevant_history=["之前买过手机"],
            user_profile={"language": "zh"},
            summary="老用户咨询",
        )
        text = ctx.to_prompt_text()
        assert "[会话摘要]" in text
        assert "[相关历史]" in text
        assert "[用户画像]" in text
        assert "[最近对话]" in text
        assert "[当前业务状态]" not in text
