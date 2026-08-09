"""Conversation-scoped distributed locking."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator


logger = logging.getLogger(__name__)


_RELEASE_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
end
return 0
"""

_RENEW_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("pexpire", KEYS[1], ARGV[2])
end
return 0
"""


class ConversationLockTimeout(TimeoutError):
    """Raised when a conversation remains busy beyond the wait timeout."""


async def _backend_call(func, *args, **kwargs):
    """Call sync Redis clients off-loop while also accepting async clients."""
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    result = await asyncio.to_thread(func, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


class RedisConversationLockManager:
    """Redis lease lock keyed by ``user_id`` and ``conv_id``.

    Each owner uses a unique token. Release and renewal are compare-and-act Lua
    operations, so an expired owner cannot delete a newer owner's lock.
    """

    PREFIX = "conversation_lock"

    def __init__(
        self,
        redis_client: Any,
        *,
        lease_s: float = 60.0,
        wait_timeout_s: float = 30.0,
        retry_interval_s: float = 0.05,
    ) -> None:
        if lease_s <= 0:
            raise ValueError("lease_s must be positive")
        if wait_timeout_s < 0:
            raise ValueError("wait_timeout_s must not be negative")
        if retry_interval_s <= 0:
            raise ValueError("retry_interval_s must be positive")
        self._redis = redis_client
        self._lease_ms = max(1, int(lease_s * 1000))
        self._wait_timeout_s = wait_timeout_s
        self._retry_interval_s = retry_interval_s

    @asynccontextmanager
    async def lock(
        self,
        user_id: str,
        conv_id: str,
    ) -> AsyncGenerator[None, None]:
        key = self._key(user_id, conv_id)
        token = uuid.uuid4().hex
        deadline = time.monotonic() + self._wait_timeout_s

        while True:
            acquired = await _backend_call(
                self._redis.set,
                key,
                token,
                nx=True,
                px=self._lease_ms,
            )
            if acquired:
                break
            if time.monotonic() >= deadline:
                raise ConversationLockTimeout(
                    f"conversation lock timed out: {user_id}:{conv_id}",
                )
            await asyncio.sleep(
                min(self._retry_interval_s, max(0.0, deadline - time.monotonic())),
            )

        stop_renewal = asyncio.Event()
        renewal = asyncio.create_task(
            self._renew_lease(key, token, stop_renewal),
            name=f"renew-conversation-lock:{key}",
        )
        try:
            yield
        finally:
            stop_renewal.set()
            renewal.cancel()
            await asyncio.gather(renewal, return_exceptions=True)
            try:
                await asyncio.shield(_backend_call(
                    self._redis.eval,
                    _RELEASE_SCRIPT,
                    1,
                    key,
                    token,
                ))
            except Exception as ex:
                logger.warning("failed to release conversation lock %s: %s", key, ex)

    async def _renew_lease(
        self,
        key: str,
        token: str,
        stop: asyncio.Event,
    ) -> None:
        interval_s = max(0.01, self._lease_ms / 3000)
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval_s)
                return
            except asyncio.TimeoutError:
                renewed = await _backend_call(
                    self._redis.eval,
                    _RENEW_SCRIPT,
                    1,
                    key,
                    token,
                    self._lease_ms,
                )
                if not renewed:
                    return

    @classmethod
    def _key(cls, user_id: str, conv_id: str) -> str:
        identity = json.dumps(
            [str(user_id), str(conv_id)],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return f"{cls.PREFIX}:{digest}"
