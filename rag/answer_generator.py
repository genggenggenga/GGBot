"""Evidence assembly and grounded answer generation for knowledge RAG."""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict

from core.prompts.rag import build_answer_generation_prompt
from core.prompts.types import PromptSpec
from rag.tokenization import count_tokens, split_by_token_budget


LLMCall = Callable[[PromptSpec], Awaitable[str]]
_CITATION_PATTERN = re.compile(r"\[\d+\]")
_PROTECTED_PATTERN = re.compile(
    r"\b[A-Za-z][A-Za-z0-9._-]*\d[A-Za-z0-9._-]*\b"
    r"|\d+(?:\.\d+)?"
    r"|[零一二两三四五六七八九十百千万亿]+(?:个)?"
    r"(?:工作日|小时|分钟|天|日|周|月|年|元)",
)


class _GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    used_citations: List[str]
    sufficient_evidence: bool


@dataclass(frozen=True)
class AnswerGenerationResult:
    response: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    sufficient_evidence: bool = False
    generated: bool = False
    fallback_reason: Optional[str] = None


class RAGAnswerGenerator:
    """Build a bounded evidence context and generate a citation-grounded answer."""

    def __init__(
        self,
        llm_call: LLMCall,
        *,
        structured_client: Any = None,
        max_chunks: int = 5,
        max_context_tokens: int = 1800,
        timeout_s: float = 8.0,
    ) -> None:
        if max_chunks < 1:
            raise ValueError("max_chunks must be positive")
        if max_context_tokens < 1:
            raise ValueError("max_context_tokens must be positive")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._llm_call = llm_call
        self._structured_client = structured_client
        self._max_chunks = max_chunks
        self._max_context_tokens = max_context_tokens
        self._timeout_s = timeout_s

    async def generate(
        self,
        question: str,
        retrieval_query: str,
        hits: Sequence[Dict[str, Any]],
        citations: Sequence[Dict[str, Any]],
    ) -> AnswerGenerationResult:
        evidence, citation_map = self.assemble_evidence(hits, citations)
        if not evidence:
            return AnswerGenerationResult(fallback_reason="no_evidence")

        try:
            prompt = build_answer_generation_prompt(
                question,
                retrieval_query,
                evidence,
            )
            if self._structured_client is not None:
                generated = await asyncio.wait_for(
                    self._structured_client.generate(
                        prompt,
                        _GeneratedAnswer,
                        tool_name="submit_grounded_answer",
                        max_tokens=768,
                        temperature=0.0,
                    ),
                    timeout=self._timeout_s,
                )
            else:
                raw = await asyncio.wait_for(
                    self._llm_call(prompt),
                    timeout=self._timeout_s,
                )
                generated = _GeneratedAnswer.model_validate(
                    self._parse_json(raw),
                )
            if not generated.sufficient_evidence:
                return AnswerGenerationResult(
                    sufficient_evidence=False,
                    generated=True,
                )
            self._validate(generated, question, evidence, citation_map)
            used = set(generated.used_citations)
            selected_citations = [
                citation_map[item["citation_id"]]
                for item in evidence
                if item["citation_id"] in used
            ]
            return AnswerGenerationResult(
                response=generated.answer.strip(),
                citations=selected_citations,
                sufficient_evidence=True,
                generated=True,
            )
        except asyncio.TimeoutError:
            reason = "timeout"
        except Exception as ex:
            reason = type(ex).__name__
        return AnswerGenerationResult(fallback_reason=reason)

    def assemble_evidence(
        self,
        hits: Sequence[Dict[str, Any]],
        citations: Sequence[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        citation_map = {
            str(item.get("citation_id")): dict(item)
            for item in citations
            if item.get("citation_id")
        }
        evidence: List[Dict[str, Any]] = []
        seen_chunks: set[str] = set()
        seen_content: set[str] = set()
        remaining = self._max_context_tokens

        for index, hit in enumerate(hits, start=1):
            if len(evidence) >= self._max_chunks or remaining <= 0:
                break
            chunk = hit.get("chunk") if isinstance(hit, dict) else None
            chunk = chunk if isinstance(chunk, dict) else {}
            content = str(
                chunk.get("content")
                or (hit.get("content") if isinstance(hit, dict) else "")
                or ""
            ).strip()
            if not content:
                continue
            chunk_id = str(chunk.get("chunk_id") or "")
            normalized = re.sub(r"\s+", " ", content).strip().lower()
            if (chunk_id and chunk_id in seen_chunks) or normalized in seen_content:
                continue

            if count_tokens(content) > remaining:
                pieces = split_by_token_budget(content, remaining)
                content = pieces[0] if pieces else ""
            if not content:
                continue

            citation_id = f"[{index}]"
            citation = citation_map.get(citation_id, {"citation_id": citation_id})
            citation_map[citation_id] = citation
            evidence.append({
                "citation_id": citation_id,
                "content": content,
                "source": citation.get("source") or chunk.get("source"),
                "title": citation.get("title") or chunk.get("title"),
                "section": citation.get("section") or chunk.get("section"),
                "page": citation.get("page") or chunk.get("page"),
                "version": citation.get("version"),
                "effective_at": citation.get("effective_at"),
                "expires_at": citation.get("expires_at"),
            })
            remaining -= count_tokens(content)
            seen_content.add(normalized)
            if chunk_id:
                seen_chunks.add(chunk_id)
        return evidence, citation_map

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        text = str(raw).strip()
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("answer generator output must be an object")
        return data

    @staticmethod
    def _validate(
        generated: _GeneratedAnswer,
        question: str,
        evidence: Sequence[Dict[str, Any]],
        citation_map: Dict[str, Dict[str, Any]],
    ) -> None:
        answer = generated.answer.strip()
        if not answer:
            raise ValueError("grounded answer is empty")

        allowed = {item["citation_id"] for item in evidence}
        markers = set(_CITATION_PATTERN.findall(answer))
        used = set(generated.used_citations)
        if not markers or markers != used or not used.issubset(allowed):
            raise ValueError("answer citations are missing or invalid")
        if not used.issubset(citation_map):
            raise ValueError("answer references unknown citations")

        paragraphs = [line.strip() for line in answer.splitlines() if line.strip()]
        if any(not _CITATION_PATTERN.search(line) for line in paragraphs):
            raise ValueError("every answer paragraph must cite evidence")

        evidence_text = "\n".join(str(item["content"]) for item in evidence)
        corpus = f"{question}\n{evidence_text}"
        uncited_answer = _CITATION_PATTERN.sub("", answer)
        protected = set(_PROTECTED_PATTERN.findall(uncited_answer))
        if any(value not in corpus for value in protected):
            raise ValueError("answer introduced an unsupported identifier or number")
