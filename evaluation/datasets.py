"""Versioned evaluation dataset loading and validation."""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_ROOT = pathlib.Path(__file__).parent.parent
_EVAL_ROOT = _ROOT / "data" / "eval"


class EvaluationCase(BaseModel):
    """One executable evaluation case.

    Legacy fields remain optional so the original smoke fixture continues to
    run unchanged while new suites can assert workflow-level contracts.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    message: Optional[str] = None
    turns: List[str] = Field(default_factory=list)
    query: Optional[str] = None
    current_state: Optional[Dict[str, Any]] = None
    expected_intent: Optional[str] = None
    expected_slots: Dict[str, Any] = Field(default_factory=dict)
    expected_act: Optional[str] = None
    expected_tool: Optional[str] = None
    expected_params: Dict[str, Any] = Field(default_factory=dict)
    expected_status: Optional[str] = None
    expected_tool_trace: Optional[List[str]] = None
    forbidden_tools: List[str] = Field(default_factory=list)
    expected_postconditions: Dict[str, Any] = Field(default_factory=dict)
    relevant_ids: List[str] = Field(default_factory=list)
    required_evidence_ids: List[str] = Field(default_factory=list)
    must_abstain: Optional[bool] = None
    risk_level: str = "normal"
    source: str = "fixture"
    review_status: str = "approved"

    @field_validator("risk_level")
    @classmethod
    def validate_risk_level(cls, value: str) -> str:
        if value not in {"normal", "medium", "high", "critical"}:
            raise ValueError("risk_level must be normal, medium, high, or critical")
        return value

    @model_validator(mode="after")
    def validate_input(self) -> "EvaluationCase":
        if not any((self.message, self.turns, self.query)):
            raise ValueError("case requires message, turns, or query")
        return self

    @property
    def prefix(self) -> str:
        return self.id.partition("-")[0]

    def raw(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


@dataclass(frozen=True)
class EvaluationDataset:
    suite: str
    version: str
    corpus_version: Optional[str]
    cases: List[EvaluationCase]
    path: pathlib.Path
    metadata: Dict[str, Any]


_DEFAULT_SUITE_FILES = {
    "smoke": _EVAL_ROOT / "smoke" / "customer_agent_cases.json",
    "golden": _EVAL_ROOT / "golden" / "golden-v1.json",
    "bad_cases": _EVAL_ROOT / "bad_cases" / "bad-cases-v1.json",
}


def load_dataset(
    suite: str = "smoke",
    *,
    path: Optional[pathlib.Path] = None,
) -> EvaluationDataset:
    """Load a list-style legacy fixture or a versioned dataset envelope."""
    if path is None:
        path = _DEFAULT_SUITE_FILES.get(suite)
        if path is None:
            raise ValueError(f"unsupported evaluation suite: {suite!r}")
        if suite == "smoke" and not path.exists():
            path = _EVAL_ROOT / "customer_agent_cases.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_cases = payload
        metadata: Dict[str, Any] = {}
        version = "legacy-v1"
        corpus_version = None
    elif isinstance(payload, dict):
        raw_cases = payload.get("cases")
        if not isinstance(raw_cases, list):
            raise ValueError(f"dataset {path} must contain a cases array")
        base_dataset = payload.get("base_dataset")
        if base_dataset:
            base_path = (path.parent / str(base_dataset)).resolve()
            base_payload = json.loads(base_path.read_text(encoding="utf-8"))
            if not isinstance(base_payload, list):
                raise ValueError(
                    f"base dataset {base_path} must use the legacy JSON list format"
                )
            raw_cases = [*base_payload, *raw_cases]
        generated_case_set = payload.get("generated_case_set")
        if generated_case_set == "golden_v1":
            from evaluation.golden_expansion import build_golden_v1_expansion

            raw_cases = [*raw_cases, *build_golden_v1_expansion()]
        elif generated_case_set:
            raise ValueError(f"unsupported generated case set: {generated_case_set!r}")
        metadata = dict(payload.get("metadata", {}))
        version = str(payload.get("version", metadata.get("version", "v1")))
        corpus_version = payload.get("corpus_version") or metadata.get("corpus_version")
    else:
        raise ValueError(f"dataset {path} must be a JSON list or object")

    cases = [EvaluationCase.model_validate(item) for item in raw_cases]
    validate_cases(cases)
    return EvaluationDataset(
        suite=suite,
        version=version,
        corpus_version=corpus_version,
        cases=cases,
        path=path,
        metadata=metadata,
    )


def validate_cases(cases: Iterable[EvaluationCase]) -> None:
    """Reject duplicate ids and contradictory workflow assertions."""
    ids = set()
    for case in cases:
        if case.id in ids:
            raise ValueError(f"duplicate evaluation case id: {case.id}")
        ids.add(case.id)
        if case.expected_tool_trace is not None and case.expected_tool:
            if case.expected_tool_trace and case.expected_tool_trace[-1] != case.expected_tool:
                raise ValueError(
                    f"{case.id}: expected_tool must equal final expected_tool_trace item"
                )
        if set(case.forbidden_tools) & set(case.expected_tool_trace or []):
            raise ValueError(f"{case.id}: a required tool cannot be forbidden")
        if case.must_abstain and (case.relevant_ids or case.required_evidence_ids):
            raise ValueError(f"{case.id}: abstention case cannot require evidence")


def dataset_summary(dataset: EvaluationDataset) -> Dict[str, Any]:
    by_risk: Dict[str, int] = {}
    by_prefix: Dict[str, int] = {}
    for case in dataset.cases:
        by_risk[case.risk_level] = by_risk.get(case.risk_level, 0) + 1
        by_prefix[case.prefix] = by_prefix.get(case.prefix, 0) + 1
    return {
        "suite": dataset.suite,
        "version": dataset.version,
        "corpus_version": dataset.corpus_version,
        "dataset_path": str(dataset.path.relative_to(_ROOT)),
        "case_count": len(dataset.cases),
        "by_risk": by_risk,
        "by_prefix": by_prefix,
    }
