import asyncio

import pytest
from fastapi import HTTPException

from api import main as api_main
from api.main import ChatRequest
from core.conversation_lock import (
    ConversationLockTimeout,
    RedisConversationLockManager,
)
from core.customer_agent_runtime import CustomerTurnResult


class FakeRedis:
    """Async Redis subset used by the lease-lock tests."""

    def __init__(self):
        self.data = {}

    async def set(self, key, value, *, nx=False, px=None):
        del px
        if nx and key in self.data:
            return False
        self.data[key] = value
        return True

    async def eval(self, script, key_count, key, *args):
        assert key_count == 1
        token = args[0]
        if self.data.get(key) != token:
            return 0
        if "pexpire" in script:
            return 1
        self.data.pop(key, None)
        return 1


@pytest.mark.asyncio
async def test_same_conversation_is_serialized():
    manager = RedisConversationLockManager(
        FakeRedis(),
        lease_s=1,
        wait_timeout_s=1,
        retry_interval_s=0.001,
    )
    active = 0
    max_active = 0

    async def work():
        nonlocal active, max_active
        async with manager.lock("user-1", "conv-1"):
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            active -= 1

    await asyncio.gather(work(), work(), work())

    assert max_active == 1


@pytest.mark.asyncio
async def test_different_conversations_can_run_in_parallel():
    manager = RedisConversationLockManager(
        FakeRedis(),
        lease_s=1,
        wait_timeout_s=1,
        retry_interval_s=0.001,
    )
    active = 0
    both_active = asyncio.Event()

    async def work(conv_id):
        nonlocal active
        async with manager.lock("user-1", conv_id):
            active += 1
            if active == 2:
                both_active.set()
            await asyncio.wait_for(both_active.wait(), timeout=0.2)
            active -= 1

    await asyncio.gather(work("conv-1"), work("conv-2"))


@pytest.mark.asyncio
async def test_expired_owner_cannot_release_new_owner_lock():
    redis = FakeRedis()
    manager = RedisConversationLockManager(redis, lease_s=1)
    key = manager._key("user-1", "conv-1")

    async with manager.lock("user-1", "conv-1"):
        redis.data[key] = "new-owner-token"

    assert redis.data[key] == "new-owner-token"


@pytest.mark.asyncio
async def test_lock_wait_has_a_bounded_timeout():
    redis = FakeRedis()
    manager = RedisConversationLockManager(
        redis,
        lease_s=1,
        wait_timeout_s=0.01,
        retry_interval_s=0.001,
    )
    redis.data[manager._key("user-1", "conv-1")] = "busy-owner"

    with pytest.raises(ConversationLockTimeout):
        async with manager.lock("user-1", "conv-1"):
            pass


@pytest.mark.asyncio
async def test_chat_holds_lock_through_memory_writes(monkeypatch):
    stages = []
    first_started = asyncio.Event()

    class MemoryContext:
        recent_messages = []

    class FakeMemory:
        async def get_context(self, user_id, conv_id, query):
            del user_id, conv_id
            stages.append(f"get:{query}")
            return MemoryContext()

        async def add_message(self, user_id, conv_id, role, content):
            del user_id, conv_id
            stages.append(f"add:{role.value}:{content}")

        async def update_profile(self, user_id, conv_id):
            del user_id, conv_id
            stages.append("profile")

    class FakeRuntime:
        async def run(self, user_id, conv_id, message, history=None):
            del user_id, conv_id, history
            stages.append(f"runtime-start:{message}")
            if message == "first":
                first_started.set()
            await asyncio.sleep(0.02)
            stages.append(f"runtime-end:{message}")
            return CustomerTurnResult(
                trace_id=f"trace:{message}",
                response=f"response:{message}",
                intent="other",
                agent_type="knowledge",
                status="awaiting_user",
                escalated=False,
                latency_ms=20,
            )

    manager = RedisConversationLockManager(
        FakeRedis(),
        lease_s=1,
        wait_timeout_s=1,
        retry_interval_s=0.001,
    )
    monkeypatch.setattr(api_main, "_memory", FakeMemory())
    monkeypatch.setattr(api_main, "_customer_runtime", FakeRuntime())
    monkeypatch.setattr(api_main, "_conversation_locks", manager)

    first = asyncio.create_task(api_main.chat(ChatRequest(
        user_id="user-1",
        conv_id="conv-1",
        message="first",
    )))
    await asyncio.wait_for(first_started.wait(), timeout=0.2)
    second = asyncio.create_task(api_main.chat(ChatRequest(
        user_id="user-1",
        conv_id="conv-1",
        message="second",
    )))
    await asyncio.gather(first, second)

    assert (
        stages.index("add:assistant:response:first")
        < stages.index("get:second")
    )


@pytest.mark.asyncio
async def test_chat_returns_conflict_when_conversation_stays_busy(monkeypatch):
    class NeverAcquired:
        def lock(self, user_id, conv_id):
            del user_id, conv_id

            class Context:
                async def __aenter__(self):
                    raise ConversationLockTimeout

                async def __aexit__(self, exc_type, exc, traceback):
                    del exc_type, exc, traceback
                    return None

            return Context()

    monkeypatch.setattr(api_main, "_memory", object())
    monkeypatch.setattr(api_main, "_customer_runtime", object())
    monkeypatch.setattr(api_main, "_conversation_locks", NeverAcquired())

    with pytest.raises(HTTPException) as captured:
        await api_main.chat(ChatRequest(
            user_id="user-1",
            conv_id="conv-1",
            message="hello",
        ))

    assert captured.value.status_code == 409
