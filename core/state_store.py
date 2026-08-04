"""DialogueState Store: abstract protocol, Redis implementation, and in-memory
implementation for unit testing.

The store is keyed by ``user_id:conv_id`` and persists a single
DialogueState per conversation.
"""
import inspect
import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

from core.agent_models import DialogueState

logger = logging.getLogger(__name__)


class StateStore(ABC):
    """Abstract base for dialogue state persistence."""

    @abstractmethod
    async def load(self, user_id: str, conv_id: str) -> Optional[DialogueState]:
        """Load state for a conversation, or None if not found."""

    @abstractmethod
    async def save(self, user_id: str, conv_id: str, state: DialogueState) -> None:
        """Persist state for a conversation."""

    @abstractmethod
    async def delete(self, user_id: str, conv_id: str) -> None:
        """Remove stored state for a conversation."""


class InMemoryStateStore(StateStore):
    """In-memory store backed by a plain dict. Suitable for unit tests.

    Returns an isolated deep copy on load so callers cannot mutate
    the stored state.
    """

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def _key(self, user_id: str, conv_id: str) -> str:
        return f"{user_id}:{conv_id}"

    async def load(self, user_id: str, conv_id: str) -> Optional[DialogueState]:
        raw = self._data.get(self._key(user_id, conv_id))
        if raw is None:
            return None
        # model_validate_json creates a brand-new instance (isolated copy)
        return DialogueState.model_validate_json(raw)

    async def save(self, user_id: str, conv_id: str, state: DialogueState) -> None:
        self._data[self._key(user_id, conv_id)] = state.model_dump_json()

    async def delete(self, user_id: str, conv_id: str) -> None:
        self._data.pop(self._key(user_id, conv_id), None)


async def _maybe_await(result):
    """Await *result* if it is awaitable, otherwise return it directly."""
    if inspect.isawaitable(result):
        return await result
    return result


class RedisStateStore(StateStore):
    """Redis-backed state store. Each state is stored as a JSON string
    under key ``dst:{user_id}:{conv_id}`` with configurable TTL.

    Works with both sync (``redis.Redis``) and async
    (``redis.asyncio.Redis``) clients by using ``inspect.isawaitable``
    to detect whether each call returns an awaitable.
    """

    PREFIX = "dst"
    DEFAULT_TTL = 86400  # 24 hours

    def __init__(
        self,
        redis_client: object,
        ttl: int = DEFAULT_TTL,
    ) -> None:
        self._redis = redis_client
        self._ttl = ttl

    def _key(self, user_id: str, conv_id: str) -> str:
        return f"{self.PREFIX}:{user_id}:{conv_id}"

    async def load(self, user_id: str, conv_id: str) -> Optional[DialogueState]:
        key = self._key(user_id, conv_id)
        raw = await _maybe_await(self._redis.get(key))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return DialogueState.model_validate_json(raw)

    async def save(self, user_id: str, conv_id: str, state: DialogueState) -> None:
        key = self._key(user_id, conv_id)
        payload = state.model_dump_json()
        await _maybe_await(self._redis.setex(key, self._ttl, payload))

    async def delete(self, user_id: str, conv_id: str) -> None:
        key = self._key(user_id, conv_id)
        await _maybe_await(self._redis.delete(key))
