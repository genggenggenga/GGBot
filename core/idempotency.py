"""Shared idempotency storage for write tool execution."""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Awaitable, Callable, Dict


class IdempotencyConflictError(ValueError):
    """An action ID was reused with different parameters."""


class ActionInProgressError(RuntimeError):
    """Another worker is currently executing the same action."""


class PreviousActionFailedError(RuntimeError):
    """A previous execution failed and its result is retained."""


Operation = Callable[[], Awaitable[Any]]


_RESERVE_SCRIPT = """
if redis.call("exists", KEYS[1]) == 1 then
    return 0
end
redis.call(
    "hset",
    KEYS[1],
    "fingerprint", ARGV[1],
    "status", "processing",
    "owner_token", ARGV[2],
    "result", ""
)
redis.call("pexpire", KEYS[1], ARGV[3])
return 1
"""

_COMPLETE_SCRIPT = """
if redis.call("hget", KEYS[1], "owner_token") ~= ARGV[1] then
    return 0
end
if redis.call("hget", KEYS[1], "status") ~= "processing" then
    return 0
end
redis.call("hset", KEYS[1], "status", ARGV[2], "result", ARGV[3])
redis.call("pexpire", KEYS[1], ARGV[4])
return 1
"""


class RedisActionExecutionRepository:
    """Coordinate write actions across application instances using Redis."""

    PREFIX = "action_execution"

    def __init__(self, redis_client: Any, *, ttl_s: int = 86400) -> None:
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        self._redis = redis_client
        self._ttl_ms = ttl_s * 1000

    async def execute(
        self,
        tool_name: str,
        action_id: str,
        arguments: Dict[str, Any],
        operation: Operation,
    ) -> Any:
        key = f"{self.PREFIX}:{tool_name}:{action_id}"
        fingerprint = _fingerprint(arguments)
        owner_token = uuid.uuid4().hex
        acquired = await self._redis.eval(
            _RESERVE_SCRIPT,
            1,
            key,
            fingerprint,
            owner_token,
            self._ttl_ms,
        )
        if not acquired:
            return await self._existing_result(key, fingerprint)

        try:
            result = await operation()
        except Exception as ex:
            await self._complete(
                key,
                owner_token,
                "failed",
                {"error": type(ex).__name__},
            )
            raise
        await self._complete(key, owner_token, "succeeded", result)
        return result

    async def _existing_result(self, key: str, fingerprint: str) -> Any:
        record = await self._redis.hgetall(key)
        if not record:
            raise ActionInProgressError("idempotency record disappeared")
        if record.get("fingerprint") != fingerprint:
            raise IdempotencyConflictError("idempotency_conflict")
        status = record.get("status")
        if status == "succeeded":
            result = json.loads(record.get("result") or "null")
            if isinstance(result, dict):
                return {**result, "idempotent_replay": True}
            return result
        if status == "failed":
            raise PreviousActionFailedError("previous_action_failed")
        raise ActionInProgressError("action_in_progress")

    async def _complete(
        self,
        key: str,
        owner_token: str,
        status: str,
        result: Any,
    ) -> None:
        encoded = json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        updated = await self._redis.eval(
            _COMPLETE_SCRIPT,
            1,
            key,
            owner_token,
            status,
            encoded,
            self._ttl_ms,
        )
        if not updated:
            raise RuntimeError("idempotency ownership lost")


class InMemoryActionExecutionRepository:
    """Deterministic repository for tests and local assembly."""

    def __init__(self) -> None:
        self._records: Dict[str, Dict[str, Any]] = {}

    async def execute(
        self,
        tool_name: str,
        action_id: str,
        arguments: Dict[str, Any],
        operation: Operation,
    ) -> Any:
        key = f"{tool_name}:{action_id}"
        fingerprint = _fingerprint(arguments)
        existing = self._records.get(key)
        if existing is not None:
            if existing["fingerprint"] != fingerprint:
                raise IdempotencyConflictError("idempotency_conflict")
            if existing["status"] == "succeeded":
                result = existing["result"]
                if isinstance(result, dict):
                    return {**result, "idempotent_replay": True}
                return result
            if existing["status"] == "failed":
                raise PreviousActionFailedError("previous_action_failed")
            raise ActionInProgressError("action_in_progress")

        self._records[key] = {
            "fingerprint": fingerprint,
            "status": "processing",
            "result": None,
        }
        try:
            result = await operation()
        except Exception:
            self._records[key]["status"] = "failed"
            raise
        self._records[key].update(status="succeeded", result=result)
        return result


def _fingerprint(arguments: Dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in arguments.items()
        if key != "action_id"
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
