"""Grounded, fail-open polishing for final customer responses."""
from __future__ import annotations

import asyncio
import json
import re
import time
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from core.prompts.response import build_prompt
from core.prompts.types import PromptSpec


LLMCall = Callable[[PromptSpec], Awaitable[str]]

_PROTECTED_FIELDS = frozenset({
    "amount",
    "currency",
    "current_status",
    "order_id",
    "refund_id",
    "status",
    "ticket_id",
    "tracking_no",
})
_IDENTIFIER_PATTERN = re.compile(
    r"\b[A-Za-z]{2,}(?:[-_][A-Za-z0-9]+|\d[A-Za-z0-9_-]*)+\b",
)
_NUMBER_PATTERN = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
_COMPLETION_MARKERS = (
    "已退款",
    "已取消",
    "已创建",
    "已完成",
    "已提交",
)


class ResponseKind(str, Enum):
    RAG = "rag"
    MULTI_AGENT = "multi_agent"
    SIMPLE_FACT = "simple_fact"
    WRITE_CONFIRMATION = "write_confirmation"
    WRITE_RESULT = "write_result"
    CLARIFICATION = "clarification"
    FAILURE = "failure"


class PolishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_response: str = Field(min_length=1)
    response_kind: ResponseKind
    protected_facts: Dict[str, str] = Field(default_factory=dict)
    required_citations: list[str] = Field(default_factory=list)


class PolishResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: str = Field(min_length=1)
    changed_meaning: bool = False


class PolishOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: str
    applied: bool = False
    fallback: bool = False
    latency_ms: float = Field(default=0.0, ge=0.0)
    validation_error: Optional[str] = None


class ResponseValidationError(ValueError):
    """Stable validation failure that is safe to expose in metrics."""


class ResponsePolisher:
    """Polish eligible responses and return the original on every failure."""

    _ELIGIBLE_KINDS = frozenset({
        ResponseKind.RAG,
        ResponseKind.MULTI_AGENT,
    })

    def __init__(
        self,
        llm_call: LLMCall,
        *,
        structured_client: Any = None,
        enabled: bool = True,
        min_chars: int = 120,
        timeout_s: float = 3.0,
    ) -> None:
        if min_chars < 0:
            raise ValueError("min_chars must not be negative")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._llm_call = llm_call
        self._structured_client = structured_client
        self._enabled = enabled
        self._min_chars = min_chars
        self._timeout_s = timeout_s

    def should_polish(self, request: PolishRequest) -> bool:
        return (
            self._enabled
            and request.response_kind in self._ELIGIBLE_KINDS
            and len(request.original_response) >= self._min_chars
        )

    async def polish(self, request: PolishRequest) -> PolishOutcome:
        if not self.should_polish(request):
            return PolishOutcome(response=request.original_response)

        started = time.monotonic()
        try:
            prompt = build_prompt(
                original_response=request.original_response,
                response_kind=request.response_kind.value,
                protected_facts=request.protected_facts,
                required_citations=request.required_citations,
            )
            if self._structured_client is not None:
                result = await asyncio.wait_for(
                    self._structured_client.generate(
                        prompt,
                        PolishResult,
                        tool_name="submit_polished_response",
                        max_tokens=512,
                        temperature=0.0,
                    ),
                    timeout=self._timeout_s,
                )
            else:
                raw = await asyncio.wait_for(
                    self._llm_call(prompt),
                    timeout=self._timeout_s,
                )
                result = PolishResult.model_validate(self._parse_json(raw))
            self._validate(request, result)
            return PolishOutcome(
                response=result.response,
                applied=True,
                latency_ms=(time.monotonic() - started) * 1000,
            )
        except asyncio.TimeoutError:
            error = "timeout"
        except ResponseValidationError as ex:
            error = str(ex)
        except (json.JSONDecodeError, ValueError, TypeError):
            error = "invalid_output"
        except Exception:
            error = "llm_error"
        return PolishOutcome(
            response=request.original_response,
            applied=True,
            fallback=True,
            latency_ms=(time.monotonic() - started) * 1000,
            validation_error=error,
        )

    @staticmethod
    def build_request(
        *,
        response: str,
        response_kind: ResponseKind,
        observations: Sequence[Any],
        citations: Sequence[Dict[str, Any]],
    ) -> PolishRequest:
        protected_facts: Dict[str, str] = {}
        for observation in observations:
            data = (
                observation.data
                if hasattr(observation, "data")
                else observation.get("data")
            )
            ResponsePolisher._collect_facts(
                data,
                response,
                protected_facts,
            )
        required_citations = list(dict.fromkeys(
            str(citation.get("citation_id", "")).strip()
            for citation in citations
            if str(citation.get("citation_id", "")).strip()
        ))
        return PolishRequest(
            original_response=response,
            response_kind=response_kind,
            protected_facts=protected_facts,
            required_citations=required_citations,
        )

    @staticmethod
    def classify(
        results: Sequence[Any],
        citations: Sequence[Dict[str, Any]],
    ) -> ResponseKind:
        if citations:
            return ResponseKind.RAG
        agents = {
            str(result.agent)
            for result in results
            if getattr(result, "agent", None)
        }
        if len(agents) > 1:
            return ResponseKind.MULTI_AGENT
        return ResponseKind.SIMPLE_FACT

    @staticmethod
    def _collect_facts(
        value: Any,
        original_response: str,
        protected_facts: Dict[str, str],
    ) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in _PROTECTED_FIELDS and item is not None:
                    rendered = str(item)
                    if rendered and rendered in original_response:
                        fact_key = key
                        suffix = 2
                        while (
                            fact_key in protected_facts
                            and protected_facts[fact_key] != rendered
                        ):
                            fact_key = f"{key}_{suffix}"
                            suffix += 1
                        protected_facts[fact_key] = rendered
                elif isinstance(item, (dict, list, tuple)):
                    ResponsePolisher._collect_facts(
                        item,
                        original_response,
                        protected_facts,
                    )
        elif isinstance(value, (list, tuple)):
            for item in value:
                ResponsePolisher._collect_facts(
                    item,
                    original_response,
                    protected_facts,
                )

    @staticmethod
    def _validate(request: PolishRequest, result: PolishResult) -> None:
        if result.changed_meaning:
            raise ResponseValidationError("changed_meaning")
        for value in request.protected_facts.values():
            if value not in result.response:
                raise ResponseValidationError("protected_fact_missing")
        for citation in request.required_citations:
            if citation not in result.response:
                raise ResponseValidationError("citation_missing")

        original_literals = ResponsePolisher._factual_literals(
            request.original_response,
        )
        polished_literals = ResponsePolisher._factual_literals(result.response)
        if not polished_literals.issubset(original_literals):
            raise ResponseValidationError("new_factual_literal")
        for marker in _COMPLETION_MARKERS:
            if marker in result.response and marker not in request.original_response:
                raise ResponseValidationError("execution_stage_changed")

    @staticmethod
    def _factual_literals(text: str) -> set[str]:
        identifiers = {
            value.upper()
            for value in _IDENTIFIER_PATTERN.findall(text)
        }
        numbers = set(_NUMBER_PATTERN.findall(text))
        return identifiers | numbers

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        text = str(raw).strip()
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("response polisher output must be an object")
        return data
