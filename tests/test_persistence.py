import sqlite3

import pytest

from core.agent_models import DialogueState, ExecutionState, TurnContext, TurnEvent, TurnEventType
from core.persistence import ConversationOwnershipError, SQLitePersistence
from core.state_store import InMemoryStateStore
from core.trace_store import TraceStore
from core.turn_engine import TurnEngine


def test_sqlite_persistence_tracks_conversation_messages_and_owner(tmp_path):
    store = SQLitePersistence(tmp_path / "ggbot.sqlite3")

    store.ensure_conversation("user-1", "conv-1", title="我要退款")
    inserted = store.record_message(
        "user-1",
        "conv-1",
        "user",
        "我要退款",
        message_id="msg-1",
        idempotency_key="idem-1",
    )
    duplicate = store.record_message(
        "user-1",
        "conv-1",
        "user",
        "我要退款",
        message_id="msg-dup",
        idempotency_key="idem-1",
    )

    assert inserted is True
    assert duplicate is False
    assert store.get_active_conversation("user-1") == "conv-1"
    with pytest.raises(ConversationOwnershipError):
        store.ensure_conversation("user-2", "conv-1")

    rows = store._conn.execute(
        "SELECT role, content FROM conversation_messages"
    ).fetchall()
    assert [(row["role"], row["content"]) for row in rows] == [
        ("user", "我要退款"),
    ]
    store.close()


@pytest.mark.asyncio
async def test_turn_engine_appends_durable_checkpoints(tmp_path):
    persistence = SQLitePersistence(tmp_path / "ggbot.sqlite3")
    state_store = InMemoryStateStore()
    engine = TurnEngine(state_store, checkpoint_store=persistence)
    engine.register(
        ExecutionState.UNDERSTANDING,
        lambda _: TurnEvent(type=TurnEventType.UNDERSTANDING_ACCEPTED),
    )
    engine.register(
        ExecutionState.ROUTING,
        lambda _: TurnEvent(type=TurnEventType.ROUTED_TO_KNOWLEDGE),
    )
    engine.register(
        ExecutionState.RETRIEVING,
        lambda _: TurnEvent(type=TurnEventType.AGENT_COMPLETED, response="完成"),
    )
    engine.register(
        ExecutionState.RESPONDING,
        lambda _: TurnEvent(type=TurnEventType.RESPONSE_READY, response="完成"),
    )

    await engine.run(
        TurnContext(
            user_id="user-1",
            conv_id="conv-1",
            trace_id="trace-1",
            dialogue_state=DialogueState(active_intent="query"),
        )
    )

    conn = sqlite3.connect(tmp_path / "ggbot.sqlite3")
    count = conn.execute("SELECT COUNT(*) FROM turn_checkpoints").fetchone()[0]
    last = conn.execute(
        """
        SELECT trace_id, execution_state FROM turn_checkpoints
        ORDER BY created_at DESC
        LIMIT 1
        """
    ).fetchone()

    assert count == 4
    assert last == ("trace-1", "completed")
    conn.close()
    persistence.close()


def test_trace_store_reads_back_from_durable_backend(tmp_path):
    persistence = SQLitePersistence(tmp_path / "ggbot.sqlite3")
    trace_store = TraceStore(durable_store=persistence)

    trace_store.append(
        "trace-1",
        {"event": "turn_end", "status": "completed", "message": "raw text"},
        user_id="user-1",
        conv_id="conv-1",
    )
    restored = TraceStore(durable_store=persistence).get("trace-1")

    assert restored[0]["event"] == "turn_end"
    assert restored[0]["status"] == "completed"
    assert "message" not in restored[0]
    persistence.close()
