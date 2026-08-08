"""Knowledge version metadata and retrieval-time visibility rules."""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, Optional


MAX_TIMESTAMP = 253402300799.0


class KnowledgeStatus(str, Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    REVOKED = "revoked"


@dataclass(frozen=True)
class RetrievalFilter:
    """Hard filters applied consistently to dense and sparse retrieval."""

    as_of: float
    knowledge_ids: tuple[str, ...] = ()

    @classmethod
    def current(
        cls,
        *,
        as_of: Optional[float | str | datetime] = None,
        knowledge_ids: Iterable[str] = (),
    ) -> "RetrievalFilter":
        return cls(
            as_of=parse_timestamp(as_of, default=time.time()),
            knowledge_ids=tuple(dict.fromkeys(knowledge_ids)),
        )

    def to_chroma_where(self) -> Dict[str, Any]:
        conditions: list[Dict[str, Any]] = [
            {"status": {"$eq": KnowledgeStatus.PUBLISHED.value}},
            {"effective_at": {"$lte": self.as_of}},
            {"expires_at": {"$gt": self.as_of}},
        ]
        if self.knowledge_ids:
            conditions.append({
                "knowledge_id": {"$in": list(self.knowledge_ids)},
            })
        return {"$and": conditions}


def normalize_knowledge_metadata(
    metadata: Optional[Dict[str, Any]],
    *,
    source: str,
    title: str,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Return complete, Chroma-compatible metadata for one knowledge version."""
    values = dict(metadata or {})
    created_at = float(now if now is not None else time.time())
    knowledge_id = str(
        values.get("knowledge_id")
        or _default_knowledge_id(source, title)
    ).strip()
    version = str(values.get("version") or "v1").strip()
    version_id = str(
        values.get("version_id")
        or f"{knowledge_id}:{version}"
    ).strip()
    raw_status = values.get("status") or KnowledgeStatus.PUBLISHED.value
    status_value = (
        raw_status.value
        if isinstance(raw_status, KnowledgeStatus)
        else str(raw_status).lower()
    )
    try:
        status = KnowledgeStatus(status_value)
    except ValueError as ex:
        raise ValueError(f"unsupported knowledge status: {status_value}") from ex

    effective_at = parse_timestamp(
        values.get("effective_at"),
        default=created_at,
    )
    expires_at = parse_timestamp(
        values.get("expires_at"),
        default=MAX_TIMESTAMP,
    )
    if expires_at <= effective_at:
        raise ValueError("expires_at must be later than effective_at")

    is_current = bool(values.get("is_current", True))
    if status != KnowledgeStatus.PUBLISHED:
        is_current = False
    values.update({
        "knowledge_id": knowledge_id,
        "version_id": version_id,
        "version": version,
        "version_seq": int(
            values.get("version_seq")
            or version_sequence(version)
        ),
        "status": status.value,
        "effective_at": effective_at,
        "expires_at": expires_at,
        "is_current": is_current,
        "created_at": parse_timestamp(
            values.get("created_at"),
            default=created_at,
        ),
        "declared_expires_at": parse_timestamp(
            values.get("declared_expires_at"),
            default=expires_at,
        ),
    })
    return values


def is_metadata_visible(
    metadata: Dict[str, Any],
    retrieval_filter: RetrievalFilter,
) -> bool:
    """Check publication, current-version, time-window and ID constraints."""
    normalized = normalize_knowledge_metadata(
        metadata,
        source=str(metadata.get("source", "legacy")),
        title=str(metadata.get("title", "")),
        now=0.0,
    )
    if normalized["status"] != KnowledgeStatus.PUBLISHED.value:
        return False
    if not (
        normalized["effective_at"]
        <= retrieval_filter.as_of
        < normalized["expires_at"]
    ):
        return False
    return (
        not retrieval_filter.knowledge_ids
        or normalized["knowledge_id"] in retrieval_filter.knowledge_ids
    )


def parse_timestamp(
    value: Optional[float | int | str | datetime],
    *,
    default: float,
) -> float:
    if value is None or value == "":
        return float(default)
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid timestamp")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        current = value
    elif isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
            return float(text)
        current = datetime.fromisoformat(text.replace("Z", "+00:00"))
    else:
        raise ValueError(f"unsupported timestamp type: {type(value).__name__}")
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).timestamp()


def version_sequence(version: str) -> int:
    """Create a deterministic sortable integer from common version strings."""
    numbers = [int(value) for value in re.findall(r"\d+", version)[:3]]
    numbers.extend([0] * (3 - len(numbers)))
    return numbers[0] * 1_000_000 + numbers[1] * 1_000 + numbers[2]


def _default_knowledge_id(source: str, title: str) -> str:
    identity = f"{source}|{title}".encode("utf-8")
    return f"knowledge-{hashlib.sha256(identity).hexdigest()[:16]}"
