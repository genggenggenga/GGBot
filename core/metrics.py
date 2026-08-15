"""Prometheus metrics emitted at the customer runtime boundaries.

Metric labels are intentionally bounded business categories. Never add user,
conversation, trace, order, or free-form error identifiers as labels.
"""
from prometheus_client import Counter, Histogram


CHAT_TURNS_TOTAL = Counter(
    "ggbot_chat_turns_total",
    "Completed customer chat turns.",
    ("agent", "intent", "status"),
)
CHAT_TURN_LATENCY_SECONDS = Histogram(
    "ggbot_chat_turn_latency_seconds",
    "End-to-end customer chat turn latency in seconds.",
    ("agent", "intent", "status"),
)
TOOL_CALLS_TOTAL = Counter(
    "ggbot_tool_calls_total",
    "Tool dispatch attempts.",
    ("agent", "tool", "tool_type", "outcome", "cached"),
)
TOOL_CALL_LATENCY_SECONDS = Histogram(
    "ggbot_tool_call_latency_seconds",
    "Tool dispatch latency in seconds.",
    ("agent", "tool", "tool_type", "cached"),
)


def record_chat_turn(
    *,
    agent: str,
    intent: str,
    status: str,
    latency_ms: float,
) -> None:
    labels = {
        "agent": agent or "unknown",
        "intent": intent or "other",
        "status": status or "unknown",
    }
    CHAT_TURNS_TOTAL.labels(**labels).inc()
    CHAT_TURN_LATENCY_SECONDS.labels(**labels).observe(max(latency_ms, 0.0) / 1000)


def record_tool_call(
    *,
    agent: str,
    tool: str,
    tool_type: str,
    success: bool,
    cached: bool,
    latency_ms: float,
) -> None:
    labels = {
        "agent": agent or "unknown",
        "tool": tool or "unknown",
        "tool_type": tool_type,
        "outcome": "success" if success else "error",
        "cached": str(cached).lower(),
    }
    TOOL_CALLS_TOTAL.labels(**labels).inc()
    TOOL_CALL_LATENCY_SECONDS.labels(
        agent=labels["agent"],
        tool=labels["tool"],
        tool_type=labels["tool_type"],
        cached=labels["cached"],
    ).observe(max(latency_ms, 0.0) / 1000)
