"""Deterministic component and task metrics for the Agent evaluation suite."""
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence


def slot_f1(
    expected: Sequence[Dict[str, Any]],
    predicted: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    true_positive = false_positive = false_negative = 0
    for expected_slots, predicted_slots in zip(expected, predicted):
        expected_items = set(expected_slots.items())
        predicted_items = set(predicted_slots.items())
        true_positive += len(expected_items & predicted_items)
        false_positive += len(predicted_items - expected_items)
        false_negative += len(expected_items - predicted_items)
    precision = _divide(true_positive, true_positive + false_positive)
    recall = _divide(true_positive, true_positive + false_negative)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _divide(2 * precision * recall, precision + recall),
    }


def joint_goal_accuracy(
    expected: Sequence[Dict[str, Any]],
    predicted: Sequence[Dict[str, Any]],
) -> float:
    if not expected:
        return 0.0
    matches = sum(
        expected_state == predicted_state
        for expected_state, predicted_state in zip(expected, predicted)
    )
    return matches / len(expected)


def recall_at_k(
    relevant_ids: Sequence[set[str]],
    ranked_ids: Sequence[Sequence[str]],
    k: int,
) -> float:
    if not relevant_ids:
        return 0.0
    values = []
    for relevant, ranked in zip(relevant_ids, ranked_ids):
        values.append(_divide(len(relevant & set(ranked[:k])), len(relevant)))
    return sum(values) / len(values)


def mean_reciprocal_rank(
    relevant_ids: Sequence[set[str]],
    ranked_ids: Sequence[Sequence[str]],
) -> float:
    if not relevant_ids:
        return 0.0
    values: List[float] = []
    for relevant, ranked in zip(relevant_ids, ranked_ids):
        rank = next(
            (index for index, item in enumerate(ranked, start=1) if item in relevant),
            None,
        )
        values.append(1.0 / rank if rank else 0.0)
    return sum(values) / len(values)


def tool_call_accuracy(cases: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    cases = list(cases)
    if not cases:
        return {"selection_accuracy": 0.0, "parameter_accuracy": 0.0}
    selected = parameters = 0
    for case in cases:
        if case.get("expected_tool") == case.get("predicted_tool"):
            selected += 1
        if case.get("expected_params") == case.get("predicted_params"):
            parameters += 1
    return {
        "selection_accuracy": selected / len(cases),
        "parameter_accuracy": parameters / len(cases),
    }


def task_completion_rate(cases: Iterable[Dict[str, Any]]) -> float:
    cases = list(cases)
    return _divide(
        sum(bool(case.get("completed")) for case in cases),
        len(cases),
    )


def citation_precision(cases: Iterable[Dict[str, Any]]) -> float:
    cases = list(cases)
    cited = sum(len(case.get("citations", [])) for case in cases)
    supported = sum(
        sum(bool(item.get("supported")) for item in case.get("citations", []))
        for case in cases
    )
    return _divide(supported, cited)


def faithfulness_rate(cases: Iterable[Dict[str, Any]]) -> float:
    cases = list(cases)
    return _divide(
        sum(bool(case.get("grounded")) for case in cases),
        len(cases),
    )


def compare_retrievers(
    cases: Iterable[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case["mode"])].append(case)
    report = {}
    for mode, values in grouped.items():
        relevant = [set(value["relevant_ids"]) for value in values]
        ranked = [value["ranked_ids"] for value in values]
        report[mode] = {
            "recall_at_5": recall_at_k(relevant, ranked, 5),
            "mrr": mean_reciprocal_rank(relevant, ranked),
        }
    return report


def _divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
