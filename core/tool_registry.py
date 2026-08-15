"""Unified ToolRegistry with ToolSpec, ToolAdapter protocol, and local adapter.

Replaces the ad-hoc Tool/MCPToolManager in mcp/tool_manager.py with a
structured, agent-aware tool execution layer.  The old module is preserved
for backward compatibility; new code should import from here.

Key design:
  - ToolSpec describes a tool's contract (schema, read/write, timeout).
  - ToolAdapter is a Protocol; LocalToolAdapter wraps a local async handler.
  - ToolRegistry owns specs, adapters, agent whitelists, and the
    confirmation gate for write operations.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field

from core.agent_models import Observation
from core.metrics import record_tool_call

logger = logging.getLogger(__name__)


# ── Tool type classification ─────────────────────────────────────────────────

class ToolType(str, Enum):
    READ = "read"
    WRITE = "write"


# ── ToolSpec ─────────────────────────────────────────────────────────────────

class ToolSpec(BaseModel):
    """Declarative tool contract."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Optional[Dict[str, Any]] = None
    tool_type: ToolType = ToolType.READ
    timeout_s: float = Field(default=30.0, gt=0)
    cache_ttl: float = Field(default=0.0, ge=0)
    supports_rerank: bool = False


# ── ToolResult ───────────────────────────────────────────────────────────────

class ToolResult(BaseModel):
    """Structured result from a tool invocation."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    data: Any = None
    tool_name: str = Field(min_length=1)
    error: Optional[str] = None
    cached: bool = False
    latency_ms: float = Field(default=0.0, ge=0)
    reranked: bool = False

    def to_observation(self) -> Observation:
        """Convert to an Observation for the dialogue state tracker."""
        return Observation(
            source="tool",
            name=self.tool_name,
            success=self.success,
            data=self.data,
            error=self.error,
        )


# ── Circuit breaker ──────────────────────────────────────────────────────────

class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Three-state circuit breaker: CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_s: float = 60.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_s = recovery_s
        self.state: CircuitState = CircuitState.CLOSED
        self.fail_count: int = 0
        self.opened_at: Optional[float] = None

    def allow(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if self.opened_at is not None and time.monotonic() - self.opened_at >= self.recovery_s:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True  # HALF_OPEN: let one probe through

    def record_success(self) -> None:
        self.fail_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self.fail_count += 1
        if self.fail_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.opened_at = time.monotonic()


# ── ToolStats ────────────────────────────────────────────────────────────────

class ToolStats:
    """Runtime statistics for a single tool."""

    __slots__ = ("total", "success", "failed", "total_latency_ms", "consecutive_fails")

    def __init__(self) -> None:
        self.total: int = 0
        self.success: int = 0
        self.failed: int = 0
        self.total_latency_ms: float = 0.0
        self.consecutive_fails: int = 0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total else 0.0


# ── ToolAdapter protocol ─────────────────────────────────────────────────────

@runtime_checkable
class ToolAdapter(Protocol):
    """Protocol that any tool adapter must satisfy."""

    spec: ToolSpec

    async def call(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
    ) -> ToolResult: ...


# ── Parameter validation ─────────────────────────────────────────────────────

_TYPE_MAP: Dict[str, Tuple[Type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def validate_params(spec: ToolSpec, params: Dict[str, Any]) -> None:
    """Validate *params* against *spec.input_schema*.

    Raises ``ValueError`` on missing required fields or type mismatches.
    """
    schema = spec.input_schema
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    for field_name in required:
        if field_name not in params:
            raise ValueError(f"tool {spec.name} missing required parameter: {field_name}")

    for key, value in params.items():
        if key in properties:
            expected_type = properties[key].get("type")
            if expected_type and expected_type in _TYPE_MAP:
                valid = isinstance(value, _TYPE_MAP[expected_type])
                if expected_type in {"number", "integer"} and isinstance(value, bool):
                    valid = False
                if not valid:
                    raise ValueError(
                        f"tool {spec.name} parameter {key} type error: "
                        f"expected {expected_type}, got {type(value).__name__}"
                    )

            allowed_values = properties[key].get("enum")
            if allowed_values is not None and value not in allowed_values:
                raise ValueError(
                    f"tool {spec.name} parameter {key} must be one of {allowed_values}"
                )


# ── LocalToolAdapter ─────────────────────────────────────────────────────────

FallbackFn = Callable[[Dict[str, Any], Optional[Dict[str, Any]], str], Any]


class LocalToolAdapter:
    """Wraps a local async handler with the full execution chain:

    schema validation -> circuit breaker -> execute (with timeout)
    -> cache write -> optional rerank.

    Does **not** check agent whitelists or write-operation confirmation;
    those concerns belong to ``ToolRegistry``.
    """

    def __init__(
        self,
        spec: ToolSpec,
        handler: Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Awaitable[Any]],
        *,
        fallback: Optional[FallbackFn] = None,
        circuit_failure_threshold: int = 5,
        circuit_recovery_s: float = 60.0,
    ) -> None:
        self.spec = spec
        self._handler = handler
        self._fallback = fallback
        self.breaker = CircuitBreaker(
            failure_threshold=circuit_failure_threshold,
            recovery_s=circuit_recovery_s,
        )
        self.stats = ToolStats()
        self._cache: Dict[str, Tuple[Any, float, bool]] = {}

    # ── core call ─────────────────────────────────────────────────────────────

    async def call(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
        rerank_top_k: int = 0,
    ) -> ToolResult:
        name = self.spec.name

        # Cache hit
        if use_cache and self.spec.cache_ttl > 0:
            cached = self._get_cache(params, rerank_top_k)
            if cached is not None:
                data, reranked = cached
                self.stats.total += 1
                self.stats.success += 1
                return ToolResult(
                    success=True,
                    data=data,
                    tool_name=name,
                    cached=True,
                    reranked=reranked,
                )

        # Circuit breaker
        if not self.breaker.allow():
            error = f"tool circuit open: {name}"
            return await self._fallback_result(params, context, error)

        self.stats.total += 1
        t0 = time.monotonic()

        try:
            # Schema validation
            validate_params(self.spec, params)

            # Execute with timeout
            data = await asyncio.wait_for(
                self._handler(params, context),
                timeout=self.spec.timeout_s,
            )
            latency = (time.monotonic() - t0) * 1000

            self.stats.success += 1
            self.stats.consecutive_fails = 0
            self.stats.total_latency_ms += latency
            self.breaker.record_success()

            # Cache write
            reranked = False
            if self.spec.cache_ttl > 0:
                self._set_cache(params, data, self.spec.cache_ttl, rerank_top_k, reranked)

            return ToolResult(
                success=True,
                data=data,
                tool_name=name,
                latency_ms=round(latency, 2),
                reranked=reranked,
            )

        except asyncio.TimeoutError:
            self.stats.failed += 1
            self.stats.consecutive_fails += 1
            self.breaker.record_failure()
            logger.error("tool timeout: %s (%ss)", name, self.spec.timeout_s)
            return await self._fallback_result(params, context, "execution timeout")

        except ValueError as ex:
            # Parameter validation error - no fallback, surface immediately
            self.stats.failed += 1
            self.stats.consecutive_fails += 1
            self.breaker.record_failure()
            return ToolResult(
                success=False,
                tool_name=name,
                error=str(ex),
            )

        except Exception as ex:
            self.stats.failed += 1
            self.stats.consecutive_fails += 1
            self.breaker.record_failure()
            logger.error("tool error: %s - %s", name, ex)
            return await self._fallback_result(params, context, str(ex))

    # ── fallback ──────────────────────────────────────────────────────────────

    async def _fallback_result(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        error: str,
    ) -> ToolResult:
        if self._fallback is None:
            return ToolResult(success=False, tool_name=self.spec.name, error=error)
        try:
            data = self._fallback(params, context, error)
            if asyncio.iscoroutine(data):
                data = await data
            return ToolResult(success=True, data=data, tool_name=self.spec.name, error=error)
        except Exception as ex:
            logger.error("tool fallback failed: %s - %s", self.spec.name, ex)
            return ToolResult(
                success=False,
                tool_name=self.spec.name,
                error=f"{error}; fallback failed: {ex}",
            )

    # ── cache ─────────────────────────────────────────────────────────────────

    def _cache_key(self, params: Dict[str, Any], rerank_top_k: int = 0) -> str:
        payload = {"params": params, "rerank_top_k": rerank_top_k}
        digest = hashlib.md5(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        return f"{self.spec.name}:{digest}"

    def _get_cache(
        self, params: Dict[str, Any], rerank_top_k: int = 0,
    ) -> Optional[Tuple[Any, bool]]:
        key = self._cache_key(params, rerank_top_k)
        entry = self._cache.get(key)
        if entry is None:
            return None
        data, expire_at, reranked = entry
        if time.monotonic() < expire_at:
            return data, reranked
        del self._cache[key]
        return None

    def _set_cache(
        self,
        params: Dict[str, Any],
        data: Any,
        ttl: float,
        rerank_top_k: int = 0,
        reranked: bool = False,
    ) -> None:
        if len(self._cache) >= 5000:
            for k in list(self._cache)[:1250]:
                del self._cache[k]
        key = self._cache_key(params, rerank_top_k)
        self._cache[key] = (data, time.monotonic() + ttl, reranked)

# ── Confirmation gate ─────────────────────────────────────────────────────────

class ConfirmationGate:
    """Tracks which write-action_ids have been confirmed.

    A write tool may only execute when its ``action_id`` is confirmed.
    """

    def __init__(self) -> None:
        self._confirmed: Set[str] = set()
        self._pending: Set[str] = set()

    def mark_pending(self, action_id: str) -> None:
        self._pending.add(action_id)

    def confirm(self, action_id: str) -> None:
        self._pending.discard(action_id)
        self._confirmed.add(action_id)

    def reject(self, action_id: str) -> None:
        self._pending.discard(action_id)
        self._confirmed.discard(action_id)

    def complete(self, action_id: str) -> None:
        """Release confirmation state after a successful write."""
        self.reject(action_id)

    def is_confirmed(self, action_id: str) -> bool:
        return action_id in self._confirmed

    def is_pending(self, action_id: str) -> bool:
        return action_id in self._pending


# ── ToolRegistry ─────────────────────────────────────────────────────────────

class ToolRegistry:
    """Unified tool registry with agent whitelists, read/write classification,
    and write-operation confirmation gate.

    Usage::

        registry = ToolRegistry()
        adapter = LocalToolAdapter(spec, handler)
        registry.register(adapter)
        registry.set_agent_whitelist("after_sales", {"query_order", "create_refund"})
        result = await registry.call("after_sales", "query_order", {"order_id": "..."})
    """

    def __init__(self) -> None:
        self._adapters: Dict[str, ToolAdapter] = {}
        self._specs: Dict[str, ToolSpec] = {}
        self._agent_whitelists: Dict[str, Set[str]] = defaultdict(set)
        self._confirmation_gate = ConfirmationGate()

    # ── register / unregister ─────────────────────────────────────────────────

    def register(self, adapter: ToolAdapter) -> None:
        if not isinstance(adapter, ToolAdapter):
            raise TypeError("adapter must implement ToolAdapter")
        name = adapter.spec.name
        self._adapters[name] = adapter
        self._specs[name] = adapter.spec
        logger.info("registered tool: %s", name)

    def unregister(self, name: str) -> None:
        self._adapters.pop(name, None)
        self._specs.pop(name, None)

    # ── query ─────────────────────────────────────────────────────────────────

    def get_spec(self, name: str) -> Optional[ToolSpec]:
        return self._specs.get(name)

    def list_tools(self, agent: Optional[str] = None) -> List[ToolSpec]:
        if agent is None:
            return list(self._specs.values())
        whitelist = self._agent_whitelists.get(agent, set())
        return [s for s in self._specs.values() if s.name in whitelist]

    # ── agent whitelist ───────────────────────────────────────────────────────

    def set_agent_whitelist(self, agent: str, tools: Set[str]) -> None:
        self._agent_whitelists[agent] = set(tools)

    def add_to_agent_whitelist(self, agent: str, tool_name: str) -> None:
        self._agent_whitelists[agent].add(tool_name)

    def is_tool_allowed(self, agent: str, tool_name: str) -> bool:
        whitelist = self._agent_whitelists.get(agent, set())
        return tool_name in whitelist

    # ── write-operation confirmation ──────────────────────────────────────────

    def mark_pending_action(self, action_id: str) -> None:
        self._confirmation_gate.mark_pending(action_id)

    def confirm_action(self, action_id: str) -> None:
        self._confirmation_gate.confirm(action_id)

    def reject_action(self, action_id: str) -> None:
        self._confirmation_gate.reject(action_id)

    def is_action_confirmed(self, action_id: str) -> bool:
        return self._confirmation_gate.is_confirmed(action_id)

    # ── core call ─────────────────────────────────────────────────────────────

    async def call(
        self,
        agent: str,
        tool_name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        action_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> ToolResult:
        """Execute *tool_name* on behalf of *agent*.

        Checks:
          1. Tool exists.
          2. Agent whitelist.
          3. Write-operation confirmation gate.

        Returns a failed ``ToolResult`` if any check fails; otherwise
        delegates to the ``LocalToolAdapter``.
        """
        started = time.monotonic()
        adapter = self._adapters.get(tool_name)
        spec = adapter.spec if adapter is not None else None
        result: ToolResult

        # 1. Tool exists?
        if adapter is None:
            result = ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"tool not found: {tool_name}",
            )
        # 2. Agent whitelist
        elif not self.is_tool_allowed(agent, tool_name):
            result = ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"agent {agent} not allowed to call {tool_name}",
            )
        # 3. Write-operation confirmation gate
        elif spec.tool_type == ToolType.WRITE and action_id is None:
            result = ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"write tool {tool_name} requires action_id for confirmation tracking",
            )
        elif (
            spec.tool_type == ToolType.WRITE
            and not self._confirmation_gate.is_confirmed(action_id)
        ):
            self._confirmation_gate.mark_pending(action_id)
            result = ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"write tool {tool_name} action {action_id} not confirmed",
            )
        else:
            result = await adapter.call(params, context, use_cache=use_cache)
            if (
                spec.tool_type == ToolType.WRITE
                and action_id is not None
                and result.success
            ):
                self._confirmation_gate.complete(action_id)

        record_tool_call(
            agent=agent,
            tool=tool_name,
            tool_type=(spec.tool_type.value if spec is not None else "unknown"),
            success=result.success,
            cached=result.cached,
            latency_ms=(time.monotonic() - started) * 1000,
        )
        return result

    # ── stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for name, adapter in self._adapters.items():
            s = getattr(adapter, "stats", None)
            breaker = getattr(adapter, "breaker", None)
            if s is None:
                continue
            result[name] = {
                "total": s.total,
                "success": s.success,
                "failed": s.failed,
                "success_rate": round(s.success_rate, 3),
                "avg_latency_ms": round(s.avg_latency_ms, 1),
                "consecutive_fails": s.consecutive_fails,
                "circuit_state": breaker.state.value if breaker is not None else "unsupported",
            }
        return result
