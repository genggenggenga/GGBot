"""Durable persistence primitives for conversations, checkpoints, and traces."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConversationOwnershipError(ValueError):
    """Raised when a conversation id belongs to a different user."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _observation_summary(observations: List[Any]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for obs in observations[:8]:
        if hasattr(obs, "model_dump"):
            data = obs.model_dump(mode="json")
        elif isinstance(obs, dict):
            data = dict(obs)
        else:
            continue
        summary.append({
            "source": data.get("source", ""),
            "name": data.get("name", ""),
            "success": bool(data.get("success", False)),
            "error": "operation_failed" if data.get("error") else None,
        })
    return summary


class SQLitePersistence:
    """Small SQLite-backed durable store.

    The schema is intentionally narrow: Redis and Chroma remain the hot
    execution stores, while SQLite keeps product history and audit snapshots.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;

                CREATE TABLE IF NOT EXISTS conversations (
                    conv_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    title TEXT,
                    active_intent TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    closed_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_conversations_user_status
                ON conversations (user_id, status, updated_at);

                CREATE TABLE IF NOT EXISTS conversation_messages (
                    message_id TEXT PRIMARY KEY,
                    conv_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT,
                    idempotency_key TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conv_id) REFERENCES conversations(conv_id)
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_idempotency
                ON conversation_messages (conv_id, idempotency_key);

                CREATE INDEX IF NOT EXISTS idx_messages_conv_time
                ON conversation_messages (conv_id, created_at);

                CREATE TABLE IF NOT EXISTS turn_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    conv_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    trace_id TEXT,
                    execution_state TEXT NOT NULL,
                    dialogue_state_json TEXT NOT NULL,
                    state_history_json TEXT,
                    observations_summary_json TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conv_id) REFERENCES conversations(conv_id)
                );

                CREATE INDEX IF NOT EXISTS idx_checkpoints_conv_time
                ON turn_checkpoints (conv_id, created_at);

                CREATE TABLE IF NOT EXISTS trace_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    conv_id TEXT,
                    user_id TEXT,
                    event_type TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_trace_events_trace_time
                ON trace_events (trace_id, created_at);

                CREATE INDEX IF NOT EXISTS idx_trace_events_conv_time
                ON trace_events (conv_id, created_at);

                CREATE TABLE IF NOT EXISTS user_profile_versions (
                    profile_version_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    chroma_snapshot_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    activated_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_profile_versions_user_status
                ON user_profile_versions (user_id, status, created_at);

                CREATE TABLE IF NOT EXISTS tool_executions (
                    execution_id TEXT PRIMARY KEY,
                    conv_id TEXT,
                    user_id TEXT,
                    trace_id TEXT,
                    tool_name TEXT NOT NULL,
                    action_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    params_hash TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    locked_by TEXT,
                    locked_until TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    UNIQUE (tool_name, idempotency_key)
                );

                CREATE INDEX IF NOT EXISTS idx_tool_executions_status_lock
                ON tool_executions (status, locked_until);

                CREATE INDEX IF NOT EXISTS idx_tool_executions_conv_time
                ON tool_executions (conv_id, updated_at);
                """
            )

    def ensure_conversation(
        self,
        user_id: str,
        conv_id: str,
        *,
        title: Optional[str] = None,
    ) -> None:
        now = _now()
        with self._lock:
            row = self._conn.execute(
                "SELECT user_id FROM conversations WHERE conv_id = ?",
                (conv_id,),
            ).fetchone()
            if row is not None:
                if row["user_id"] != user_id:
                    raise ConversationOwnershipError("conversation belongs to another user")
                self._conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE conv_id = ?",
                    (now, conv_id),
                )
                return
            self._conn.execute(
                """
                INSERT INTO conversations (
                    conv_id, user_id, status, title, created_at, updated_at
                ) VALUES (?, ?, 'active', ?, ?, ?)
                """,
                (conv_id, user_id, title, now, now),
            )

    def get_active_conversation(self, user_id: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT conv_id FROM conversations
                WHERE user_id = ? AND status = 'active'
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        return str(row["conv_id"]) if row else None

    def update_conversation(
        self,
        user_id: str,
        conv_id: str,
        *,
        status: Optional[str] = None,
        active_intent: Optional[str] = None,
    ) -> None:
        now = _now()
        closed_at = now if status in {"completed", "handoff", "failed"} else None
        with self._lock:
            self.ensure_conversation(user_id, conv_id)
            self._conn.execute(
                """
                UPDATE conversations
                SET status = COALESCE(?, status),
                    active_intent = COALESCE(?, active_intent),
                    updated_at = ?,
                    closed_at = COALESCE(?, closed_at)
                WHERE conv_id = ? AND user_id = ?
                """,
                (status, active_intent, now, closed_at, conv_id, user_id),
            )

    def record_message(
        self,
        user_id: str,
        conv_id: str,
        role: str,
        content: str,
        *,
        message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> bool:
        self.ensure_conversation(user_id, conv_id)
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO conversation_messages (
                    message_id, conv_id, user_id, role, content,
                    metadata_json, idempotency_key, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id or str(uuid.uuid4()),
                    conv_id,
                    user_id,
                    role,
                    content,
                    _json_dumps(metadata or {}),
                    idempotency_key,
                    _now(),
                ),
            )
        return cursor.rowcount > 0

    def record_turn_checkpoint(self, context: Any) -> None:
        dialogue_state = context.dialogue_state
        status = _checkpoint_status(_enum_value(context.execution_state))
        payload = (
            dialogue_state.model_dump(mode="json")
            if hasattr(dialogue_state, "model_dump")
            else dialogue_state
        )
        with self._lock:
            self.ensure_conversation(context.user_id, context.conv_id)
            self._conn.execute(
                """
                INSERT INTO turn_checkpoints (
                    checkpoint_id, conv_id, user_id, trace_id,
                    execution_state, dialogue_state_json, state_history_json,
                    observations_summary_json, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    context.conv_id,
                    context.user_id,
                    getattr(context, "trace_id", None),
                    _enum_value(context.execution_state),
                    _json_dumps(payload),
                    _json_dumps([_enum_value(item) for item in context.state_history]),
                    _json_dumps(_observation_summary(list(context.observations))),
                    status,
                    _now(),
                ),
            )

    def record_trace_event(
        self,
        trace_id: str,
        event: Dict[str, Any],
        *,
        user_id: Optional[str] = None,
        conv_id: Optional[str] = None,
    ) -> None:
        event_type = str(event.get("event") or "unknown")
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO trace_events (
                    trace_id, conv_id, user_id, event_type, event_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (trace_id, conv_id, user_id, event_type, _json_dumps(event), _now()),
            )

    def get_trace_events(self, trace_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT event_json FROM trace_events
                WHERE trace_id = ?
                ORDER BY id ASC
                """,
                (trace_id,),
            ).fetchall()
        return [json.loads(row["event_json"]) for row in rows]

    def activate_user_profile_version(
        self,
        user_id: str,
        chroma_snapshot_id: str,
        profile: Dict[str, Any],
        *,
        profile_version_id: Optional[str] = None,
    ) -> str:
        version_id = profile_version_id or str(uuid.uuid4())
        now = _now()
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.execute(
                    """
                    UPDATE user_profile_versions
                    SET status = 'inactive'
                    WHERE user_id = ? AND status = 'active'
                    """,
                    (user_id,),
                )
                self._conn.execute(
                    """
                    INSERT INTO user_profile_versions (
                        profile_version_id, user_id, chroma_snapshot_id,
                        status, profile_json, created_at, activated_at
                    ) VALUES (?, ?, ?, 'active', ?, ?, ?)
                    """,
                    (
                        version_id,
                        user_id,
                        chroma_snapshot_id,
                        _json_dumps(profile),
                        now,
                        now,
                    ),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return version_id

    def get_active_profile_version(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT profile_version_id, chroma_snapshot_id, profile_json
                FROM user_profile_versions
                WHERE user_id = ? AND status = 'active'
                ORDER BY activated_at DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "profile_version_id": row["profile_version_id"],
            "chroma_snapshot_id": row["chroma_snapshot_id"],
            "profile": json.loads(row["profile_json"]),
        }

    def get_tool_execution(
        self,
        tool_name: str,
        idempotency_key: str,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM tool_executions
                WHERE tool_name = ? AND idempotency_key = ?
                """,
                (tool_name, idempotency_key),
            ).fetchone()
        return _row_to_dict(row)

    def record_tool_execution_running(
        self,
        tool_name: str,
        action_id: str,
        idempotency_key: str,
        params_hash: str,
        params: Dict[str, Any],
        *,
        worker_id: str,
        locked_until: str,
        context: Optional[Dict[str, Any]] = None,
        max_attempts: int = 3,
    ) -> None:
        now = _now()
        context = context or {}
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO tool_executions (
                    execution_id, conv_id, user_id, trace_id, tool_name,
                    action_id, idempotency_key, params_hash, params_json,
                    status, attempt_count, max_attempts, locked_by,
                    locked_until, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', 1, ?, ?, ?, ?, ?)
                ON CONFLICT(tool_name, idempotency_key) DO UPDATE SET
                    status = CASE
                        WHEN tool_executions.status = 'succeeded'
                        THEN tool_executions.status ELSE 'running'
                    END,
                    attempt_count = CASE
                        WHEN tool_executions.status = 'succeeded'
                        THEN tool_executions.attempt_count
                        ELSE tool_executions.attempt_count + 1
                    END,
                    locked_by = CASE
                        WHEN tool_executions.status = 'succeeded'
                        THEN tool_executions.locked_by ELSE excluded.locked_by
                    END,
                    locked_until = CASE
                        WHEN tool_executions.status = 'succeeded'
                        THEN tool_executions.locked_until ELSE excluded.locked_until
                    END,
                    updated_at = excluded.updated_at
                """,
                (
                    str(uuid.uuid4()),
                    context.get("conv_id"),
                    context.get("user_id"),
                    context.get("trace_id"),
                    tool_name,
                    action_id,
                    idempotency_key,
                    params_hash,
                    _json_dumps(params),
                    max_attempts,
                    worker_id,
                    locked_until,
                    now,
                    now,
                ),
            )

    def complete_tool_execution(
        self,
        tool_name: str,
        idempotency_key: str,
        status: str,
        *,
        result: Any = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        now = _now()
        completed_at = now if status in {"succeeded", "failed_terminal"} else None
        with self._lock:
            self._conn.execute(
                """
                UPDATE tool_executions
                SET status = ?,
                    result_json = ?,
                    error_code = ?,
                    error_message = ?,
                    locked_by = NULL,
                    locked_until = NULL,
                    updated_at = ?,
                    completed_at = COALESCE(?, completed_at)
                WHERE tool_name = ? AND idempotency_key = ?
                """,
                (
                    status,
                    _json_dumps(result) if result is not None else None,
                    error_code,
                    error_message,
                    now,
                    completed_at,
                    tool_name,
                    idempotency_key,
                ),
            )

    def list_recoverable_tool_executions(
        self,
        *,
        now: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        cutoff = now or _now()
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM tool_executions
                WHERE status IN ('running', 'unknown', 'failed_retryable')
                  AND (locked_until IS NULL OR locked_until <= ?)
                ORDER BY updated_at ASC
                LIMIT ?
                """,
                (cutoff, limit),
            ).fetchall()
        return [_row_to_dict(row) for row in rows if row is not None]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _checkpoint_status(execution_state: str) -> str:
    if execution_state == "completed":
        return "completed"
    if execution_state == "failed":
        return "failed"
    if execution_state in {"clarifying", "awaiting_confirmation"}:
        return "awaiting_user"
    return "running"


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}
