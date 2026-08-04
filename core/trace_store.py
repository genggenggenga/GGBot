"""Minimal in-memory trace storage for the customer-agent runtime.

Observation summaries are trimmed: only source, name, success flag, and a
short data_preview (up to 120 chars) are kept.  User text, full prompts,
and hidden reasoning chains are never stored in the trace.
"""
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

# Maximum characters kept from Observation.data in a trace event.
_DATA_PREVIEW_LIMIT = 120
_SUMMARY_LIMIT = 8

# Fields that must NEVER appear in a trace event.
_FORBIDDEN_KEYS = frozenset({
    "user_message", "message", "prompt", "full_prompt",
    "system_prompt", "query", "input", "user_input", "text", "content",
    "messages", "request", "request_body",
    "hidden_reasoning", "reasoning", "reasoning_chain", "chain_of_thought",
    "raw_llm_output",
})


def _sanitize_value(value: Any) -> Any:
    """Recursively remove fields that may contain request text or reasoning."""
    if isinstance(value, dict):
        return {
            key: _sanitize_value(item)
            for key, item in value.items()
            if str(key).lower() not in _FORBIDDEN_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    return value


def _preview(value: Any) -> str:
    sanitized = _sanitize_value(value)
    try:
        rendered = json.dumps(
            sanitized,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    except (TypeError, ValueError):
        rendered = str(sanitized)
    if len(rendered) > _DATA_PREVIEW_LIMIT:
        return rendered[:_DATA_PREVIEW_LIMIT] + "..."
    return rendered


def summarize_observations(observations: Sequence[Any]) -> List[Dict[str, Any]]:
    """Produce a trimmed summary of Observation objects for trace storage.

    Only keeps source, name, success, and a truncated data_preview.
    Never includes the full observation data, user text, or hidden reasoning.
    """
    summaries: List[Dict[str, Any]] = []
    for index, obs in enumerate(observations):
        if index >= _SUMMARY_LIMIT:
            break
        # Support both Observation model instances and plain dicts.
        if hasattr(obs, "source"):
            source = obs.source
            name = obs.name
            success = obs.success
            data = obs.data
            error = obs.error
        else:
            source = obs.get("source", "")
            name = obs.get("name", "")
            success = obs.get("success", False)
            data = obs.get("data")
            error = obs.get("error")

        summary: Dict[str, Any] = {
            "source": source,
            "name": name,
            "success": success,
        }
        if data is not None:
            summary["data_preview"] = _preview(data)
        if error and not success:
            # Raw error strings can echo requests or prompts.
            summary["error_preview"] = "operation_failed"
        summaries.append(summary)
    return summaries


def _scrub_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively remove forbidden keys from a trace event dict."""
    return _sanitize_value(event)


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    event: str = Field(min_length=1)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    agent: Optional[str] = None
    success: Optional[bool] = None
    latency_ms: Optional[float] = Field(default=None, ge=0)
    result_summary: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        max_length=_SUMMARY_LIMIT,
    )
    observation_summaries: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        max_length=_SUMMARY_LIMIT,
    )


class TraceStore:
    """Store bounded, user-visible execution summaries by trace id.

    Scrubs forbidden keys (user text, full prompts, hidden reasoning)
    before persisting any event.
    """

    def __init__(self, max_traces: int = 1000) -> None:
        self._max_traces = max_traces
        self._events: Dict[str, List[Dict[str, Any]]] = {}

    def append(self, trace_id: str, event: Dict[str, Any]) -> None:
        if trace_id not in self._events and len(self._events) >= self._max_traces:
            self._events.pop(next(iter(self._events)))
        scrubbed = _scrub_event(event)
        payload = TraceEvent.model_validate(scrubbed).model_dump(mode="json")
        self._events.setdefault(trace_id, []).append(payload)

    def get(self, trace_id: str) -> List[Dict[str, Any]]:
        return [dict(event) for event in self._events.get(trace_id, [])]
