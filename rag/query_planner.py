"""Reference resolution and bounded multi-query planning for knowledge RAG."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from core.prompts.rag import build_query_planner_prompt
from core.prompts.types import PromptSpec
from rag.models import QueryPlan


logger = logging.getLogger(__name__)
LLMCall = Callable[[PromptSpec], Awaitable[str]]
_REFERENCE_PATTERN = re.compile(
    r"(它|这个|那个|上面(?:的)?|刚才(?:的)?|该(?:政策|问题|流程|功能))",
)
_PROTECTED_PATTERN = re.compile(
    r"\b[A-Za-z][A-Za-z0-9._-]*\d[A-Za-z0-9._-]*\b"
    r"|\b\d+(?:\.\d+)?\b",
)


class _QueryPlanOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    standalone_query: str
    alternative_queries: List[str] = Field(default_factory=list)
    resolved_references: Dict[str, str] = Field(default_factory=dict)
    confidence: float = 0.0


class QueryPlanner:
    """Create one standalone query and bounded retrieval alternatives."""

    def __init__(
        self,
        llm_call: Optional[LLMCall] = None,
        *,
        structured_client: Any = None,
        enabled: bool = True,
        max_queries: int = 3,
        min_confidence: float = 0.5,
    ) -> None:
        if max_queries < 1:
            raise ValueError("max_queries must be positive")
        self._llm_call = llm_call
        self._structured_client = structured_client
        self._enabled = enabled
        self.max_queries = max_queries
        self._min_confidence = min_confidence

    async def plan(
        self,
        message: str,
        *,
        history: Optional[List[Dict[str, str]]] = None,
        dialogue_state: Optional[Dict[str, Any]] = None,
    ) -> QueryPlan:
        fallback = self._fallback(message, history)
        if not self._enabled:
            return fallback.model_copy(update={"fallback_reason": "disabled"})
        if self._llm_call is None and self._structured_client is None:
            return fallback.model_copy(
                update={"fallback_reason": "llm_unavailable"},
            )

        try:
            prompt = self._build_prompt(message, history, dialogue_state)
            if self._structured_client is not None:
                output = await self._structured_client.generate(
                    prompt,
                    _QueryPlanOutput,
                    tool_name="submit_query_plan",
                    max_tokens=512,
                    temperature=0.1,
                )
                data = output.model_dump()
            else:
                raw = await self._llm_call(prompt)
                data = self._parse_json(raw)
            plan = self._validate_plan(message, data)
            if plan.confidence < self._min_confidence:
                return fallback.model_copy(
                    update={"fallback_reason": "low_confidence"},
                )
            return plan
        except Exception as ex:
            logger.warning("query planning failed, using original query: %s", ex)
            return fallback.model_copy(
                update={"fallback_reason": type(ex).__name__},
            )

    def _validate_plan(
        self,
        message: str,
        data: Dict[str, Any],
    ) -> QueryPlan:
        standalone = str(data.get("standalone_query", "")).strip()
        if not standalone:
            raise ValueError("standalone_query is required")
        alternatives = data.get("alternative_queries", [])
        if not isinstance(alternatives, list):
            raise ValueError("alternative_queries must be a list")
        alternatives = [
            str(item).strip()[:500]
            for item in alternatives
            if str(item).strip()
        ]
        resolved = data.get("resolved_references", {})
        if not isinstance(resolved, dict):
            resolved = {}
        confidence = float(data.get("confidence", 0.0))

        protected = set(_PROTECTED_PATTERN.findall(message))
        rewritten = " ".join([standalone, *alternatives])
        if any(value not in rewritten for value in protected):
            raise ValueError("protected identifier was changed or removed")

        return QueryPlan(
            original_query=message,
            standalone_query=standalone[:800],
            alternative_queries=alternatives[:self.max_queries],
            resolved_references={
                str(key)[:100]: str(value)[:300]
                for key, value in resolved.items()
                if str(key).strip() and str(value).strip()
            },
            confidence=max(0.0, min(1.0, confidence)),
            used_llm=True,
        )

    def _fallback(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> QueryPlan:
        standalone = message
        if _REFERENCE_PATTERN.search(message):
            anchor = self._last_user_message(history, message)
            if anchor:
                standalone = f"{anchor}；当前追问：{message}"
        return QueryPlan(
            original_query=message,
            standalone_query=standalone,
            confidence=0.0 if standalone != message else 1.0,
        )

    def _build_prompt(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
        dialogue_state: Optional[Dict[str, Any]],
    ) -> PromptSpec:
        return build_query_planner_prompt(
            message,
            history=history,
            dialogue_state=dialogue_state,
            max_alternatives=self.max_queries - 1,
        )

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        text = str(raw).strip()
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("query planner output must be an object")
        return data

    @staticmethod
    def _last_user_message(
        history: Optional[List[Dict[str, str]]],
        current: str,
    ) -> str:
        for item in reversed(history or []):
            content = str(item.get("content", "")).strip()
            if (
                item.get("role") == "user"
                and content
                and content != current
            ):
                return content[:800]
        return ""
