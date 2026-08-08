"""Reference resolution and bounded multi-query planning for knowledge RAG."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional

from rag.models import QueryPlan


logger = logging.getLogger(__name__)
LLMCall = Callable[[str], Awaitable[str]]
_REFERENCE_PATTERN = re.compile(
    r"(它|这个|那个|上面(?:的)?|刚才(?:的)?|该(?:政策|问题|流程|功能))",
)
_PROTECTED_PATTERN = re.compile(
    r"\b[A-Za-z][A-Za-z0-9._-]*\d[A-Za-z0-9._-]*\b"
    r"|\b\d+(?:\.\d+)?\b",
)


class QueryPlanner:
    """Create one standalone query and bounded retrieval alternatives."""

    def __init__(
        self,
        llm_call: Optional[LLMCall] = None,
        *,
        enabled: bool = True,
        max_queries: int = 3,
        min_confidence: float = 0.5,
    ) -> None:
        if max_queries < 1:
            raise ValueError("max_queries must be positive")
        self._llm_call = llm_call
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
        if self._llm_call is None:
            return fallback.model_copy(
                update={"fallback_reason": "llm_unavailable"},
            )

        try:
            raw = await self._llm_call(
                self._build_prompt(message, history, dialogue_state),
            )
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
    ) -> str:
        recent = [
            {
                "role": str(item.get("role", "user"))[:20],
                "content": str(item.get("content", ""))[:800],
            }
            for item in (history or [])[-5:]
        ]
        state = {
            key: value
            for key, value in (dialogue_state or {}).items()
            if key in {
                "active_intent",
                "slots",
                "last_agent",
                "completed_goals",
            }
        }
        return f"""你是知识库检索 Query Planner。只改写问题，不回答问题。

任务：
1. 根据最近对话和业务状态消解“它、这个、上面的、该政策”等指代。
2. 把当前问题改写为脱离对话也能理解的 standalone_query。
3. 生成最多 {max(0, self.max_queries - 1)} 条不同表达的检索查询。

约束：
- 不得创造上下文中不存在的事实、ID、数字、地区或产品。
- 必须原样保留当前问题中的订单号、错误码、金额和数字。
- 查询应简短，适合向量与 BM25 检索。
- 仅输出 JSON。

最近对话：{json.dumps(recent, ensure_ascii=False)}
业务状态：{json.dumps(state, ensure_ascii=False)}
当前问题：{json.dumps(message, ensure_ascii=False)}

输出格式：
{{
  "standalone_query": "...",
  "alternative_queries": ["...", "..."],
  "resolved_references": {{"这个": "..."}},
  "confidence": 0.0
}}"""

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        text = str(raw).strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("query planner output is not JSON")
        data = json.loads(text[start:end])
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
