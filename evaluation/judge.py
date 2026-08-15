"""Optional LLM-as-Judge for semantic answer quality evaluation."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content
from core.prompts.evaluation import build_judge_prompt


@dataclass
class JudgeInput:
    user_question: str
    candidate_response: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    tool_observations: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    expected_behavior: Optional[str] = None


@dataclass
class JudgeScores:
    relevance: float
    accuracy: float
    completeness: float
    helpfulness: float
    safety: float
    groundedness: float
    violations: List[str] = field(default_factory=list)
    judge_failed: bool = False
    error: Optional[str] = None

    @property
    def overall(self) -> float:
        return sum((
            self.relevance,
            self.accuracy,
            self.completeness,
            self.helpfulness,
            self.safety,
            self.groundedness,
        )) / 6

    def model_dump(self) -> Dict[str, Any]:
        data = asdict(self)
        data["overall"] = self.overall
        return data


class LLMJudge:
    """Evidence-constrained evaluator; never used as a deterministic gate."""

    prompt_version = "judge-v2"

    def __init__(self, client: AsyncAnthropic, model: str) -> None:
        self._client = client
        self._model = model

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
    ) -> "LLMJudge":
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return cls(AsyncAnthropic(**kwargs), model)

    async def judge(self, item: JudgeInput) -> JudgeScores:
        prompt = build_judge_prompt(
            question=_clean(item.user_question),
            response=_clean(item.candidate_response),
            context=json.dumps(
                {
                    "conversation_history": item.conversation_history,
                    "tool_observations": item.tool_observations,
                    "evidence": item.evidence,
                    "expected_behavior": item.expected_behavior,
                },
                ensure_ascii=False,
            ),
        )
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=384,
                temperature=0.0,
                system=prompt.system,
                messages=[{"role": "user", "content": prompt.user}],
            )
            data = json.loads(extract_text_content(response.content))
            return JudgeScores(
                relevance=_score(data, "relevance"),
                accuracy=_score(data, "accuracy"),
                completeness=_score(data, "completeness"),
                helpfulness=_score(data, "helpfulness"),
                safety=_score(data, "safety"),
                groundedness=_score(data, "groundedness"),
                violations=[
                    str(value) for value in data.get("violations", [])
                ],
            )
        except Exception as ex:
            return JudgeScores(
                relevance=0.0,
                accuracy=0.0,
                completeness=0.0,
                helpfulness=0.0,
                safety=0.0,
                groundedness=0.0,
                judge_failed=True,
                error=str(ex),
            )


def _score(data: Dict[str, Any], name: str) -> float:
    value = float(data.get(name, 0.0))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"judge score {name} must be in [0, 1]")
    return value


def _clean(value: Any) -> str:
    return str(value or "").encode("utf-8", errors="ignore").decode("utf-8")
