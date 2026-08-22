"""Shared idempotency storage for write tool execution."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        redis_client: Any,
        *,
        ttl_s: int = 86400,
        durable_store: Any = None,
        worker_id: str | None = None,
        max_attempts: int = 3,
    ) -> None:
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self._redis = redis_client
        self._ttl_ms = ttl_s * 1000
        self._ttl_s = ttl_s
        self._durable_store = durable_store
        self._worker_id = worker_id or uuid.uuid4().hex
        self._max_attempts = max_attempts

    async def execute(
        self,
        tool_name: str,
        action_id: str,
        arguments: Dict[str, Any],
        operation: Operation,
        *,
        context: Dict[str, Any] | None = None,
    ) -> Any:
        key = f"{self.PREFIX}:{tool_name}:{action_id}"
        fingerprint = _fingerprint(arguments)
        durable = await self._get_durable(tool_name, action_id)
        if durable is not None:
            self._validate_durable_record(durable, fingerprint)
            if durable.get("status") == "succeeded":
                return _with_replay_marker(_decode_json(durable.get("result_json")))
            if durable.get("status") in {"failed", "failed_terminal"}:
                raise PreviousActionFailedError("previous_action_failed")

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
            return await self._existing_result(
                key,
                tool_name,
                action_id,
                fingerprint,
            )

        await self._mark_durable_running(
            tool_name,
            action_id,
            fingerprint,
            arguments,
            context=context,
        )

        try:
            result = await operation()
        except Exception as ex:
            await self._complete_durable(
                tool_name,
                action_id,
                "failed_terminal",
                error_code=type(ex).__name__,
                error_message=str(ex)[:500],
            )
            await self._complete(
                key,
                owner_token,
                "failed",
                {"error": type(ex).__name__},
            )
            raise
        await self._complete_durable(
            tool_name,
            action_id,
            "succeeded",
            result=result,
        )
        await self._complete(key, owner_token, "succeeded", result)
        return result

    async def _existing_result(
        self,
        key: str,
        tool_name: str,
        action_id: str,
        fingerprint: str,
    ) -> Any:
        record = await self._redis.hgetall(key)
        if not record:
            durable = await self._get_durable(tool_name, action_id)
            if durable is None:
                raise ActionInProgressError("idempotency record disappeared")
            self._validate_durable_record(durable, fingerprint)
            if durable.get("status") == "succeeded":
                return _with_replay_marker(_decode_json(durable.get("result_json")))
            if durable.get("status") in {"failed", "failed_terminal"}:
                raise PreviousActionFailedError("previous_action_failed")
            raise ActionInProgressError("action_in_progress")
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
        durable = await self._get_durable(tool_name, action_id)
        if durable is not None:
            self._validate_durable_record(durable, fingerprint)
            if durable.get("status") == "succeeded":
                return _with_replay_marker(_decode_json(durable.get("result_json")))
            if durable.get("status") in {"failed", "failed_terminal"}:
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

    async def _get_durable(
        self,
        tool_name: str,
        idempotency_key: str,
    ) -> Dict[str, Any] | None:
        if self._durable_store is None:
            return None
        method = getattr(self._durable_store, "get_tool_execution", None)
        if method is None:
            return None
        return await _maybe_backend_call(method, tool_name, idempotency_key)

    async def _mark_durable_running(
        self,
        tool_name: str,
        action_id: str,
        fingerprint: str,
        arguments: Dict[str, Any],
        *,
        context: Dict[str, Any] | None,
    ) -> None:
        if self._durable_store is None:
            return
        method = getattr(
            self._durable_store,
            "record_tool_execution_running",
            None,
        )
        if method is None:
            return
        locked_until = (
            datetime.now(timezone.utc) + timedelta(seconds=self._ttl_s)
        ).isoformat()
        try:
            await _maybe_backend_call(
                method,
                tool_name,
                action_id,
                action_id,
                fingerprint,
                arguments,
                worker_id=self._worker_id,
                locked_until=locked_until,
                context=_tool_context(context),
                max_attempts=self._max_attempts,
            )
        except Exception as ex:
            logger.warning("记录工具执行 running 状态失败: %s", ex)

    async def _complete_durable(
        self,
        tool_name: str,
        idempotency_key: str,
        status: str,
        *,
        result: Any = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        if self._durable_store is None:
            return
        method = getattr(self._durable_store, "complete_tool_execution", None)
        if method is None:
            return
        try:
            await _maybe_backend_call(
                method,
                tool_name,
                idempotency_key,
                status,
                result=result,
                error_code=error_code,
                error_message=error_message,
            )
        except Exception as ex:
            logger.warning("记录工具执行完成状态失败: %s", ex)

    @staticmethod
    def _validate_durable_record(
        record: Dict[str, Any],
        fingerprint: str,
    ) -> None:
        if record.get("params_hash") != fingerprint:
            raise IdempotencyConflictError("idempotency_conflict")


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
        *,
        context: Dict[str, Any] | None = None,
    ) -> Any:
        del context
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


async def _maybe_backend_call(func, *args, **kwargs):
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    result = await asyncio.to_thread(func, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _decode_json(raw: Any) -> Any:
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str):
        return raw
    return json.loads(raw)


def _with_replay_marker(result: Any) -> Any:
    if isinstance(result, dict):
        return {**result, "idempotent_replay": True}
    return result


def _tool_context(context: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(context, dict):
        return {}
    return {
        key: context[key]
        for key in ("user_id", "conv_id", "trace_id")
        if context.get(key)
    }
