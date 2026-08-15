"""Query-variation matrix for the chunking Golden candidate set."""
from __future__ import annotations

from typing import Any, Dict, List


def build_chunking_v1_expansion(
    base_cases: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Create 68 labelled paraphrase variants from 12 reviewed anchors."""
    variants = (
        "请问{query}",
        "想了解{query}",
        "{query}，请说明",
        "关于{query}的规定",
        "{query}怎么办",
    )
    cases: List[Dict[str, Any]] = []
    for source_case in base_cases:
        for index, template in enumerate(variants, start=1):
            cases.append({
                **source_case,
                "id": f"{source_case['id']}-v{index}",
                "query": template.format(query=source_case["query"]),
                "source": "chunking_v1_query_variation",
                "review_status": "generated_pending_human_calibration",
            })
    for source_case in base_cases[:8]:
        cases.append({
            **source_case,
            "id": f"{source_case['id']}-v6",
            "query": f"客服如何处理{source_case['query']}",
            "source": "chunking_v1_query_variation",
            "review_status": "generated_pending_human_calibration",
        })
    if len(cases) != 68:
        raise AssertionError(f"chunking expansion must contain 68 cases, got {len(cases)}")
    return cases
