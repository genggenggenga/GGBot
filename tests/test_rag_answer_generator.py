import json

import pytest

from rag.answer_generator import RAGAnswerGenerator
from rag.tokenization import count_tokens


def _hit(chunk_id: str, content: str):
    return {
        "chunk": {
            "chunk_id": chunk_id,
            "content": content,
            "source": "policy.md",
        },
        "score": 0.9,
    }


def _citation(index: int):
    return {
        "citation_id": f"[{index}]",
        "chunk_id": f"chunk-{index}",
        "source": "policy.md",
    }


@pytest.mark.asyncio
async def test_answer_generator_combines_evidence_and_returns_used_citations():
    async def llm_call(prompt):
        payload = json.loads(prompt.user.split("\n", 1)[1])
        assert [item["citation_id"] for item in payload["evidence"]] == [
            "[1]",
            "[2]",
        ]
        return json.dumps({
            "answer": "商品签收后七天内可以申请退款。[1]\n退款将原路退回。[2]",
            "used_citations": ["[1]", "[2]"],
            "sufficient_evidence": True,
        }, ensure_ascii=False)

    generator = RAGAnswerGenerator(llm_call)
    result = await generator.generate(
        "退款期限和退款渠道是什么",
        "退款期限 退款渠道",
        [
            _hit("chunk-1", "商品签收后七天内可以申请退款。"),
            _hit("chunk-2", "退款审核通过后将原路退回。"),
        ],
        [_citation(1), _citation(2)],
    )

    assert result.generated is True
    assert result.sufficient_evidence is True
    assert result.response.count("\n") == 1
    assert [item["citation_id"] for item in result.citations] == ["[1]", "[2]"]


def test_answer_generator_deduplicates_and_enforces_context_budget():
    async def llm_call(prompt):
        raise AssertionError("not called")

    generator = RAGAnswerGenerator(
        llm_call,
        max_chunks=3,
        max_context_tokens=15,
    )
    evidence, _ = generator.assemble_evidence(
        [
            _hit("chunk-1", "退款申请需要订单号。"),
            _hit("chunk-duplicate", "退款申请需要订单号。"),
            _hit("chunk-2", "审核通过后退款原路退回到账户。"),
        ],
        [_citation(1), _citation(2), _citation(3)],
    )

    assert len(evidence) == 2
    assert sum(count_tokens(item["content"]) for item in evidence) <= 15


def test_answer_generator_includes_parent_context_in_evidence():
    async def llm_call(prompt):
        raise AssertionError("not called")

    generator = RAGAnswerGenerator(llm_call)
    hit = _hit("chunk-1", "第一步：核验订单状态。")
    hit["parent_context"] = {
        "chunk_id": "parent-1",
        "content": "完整退款流程：核验订单，确认动作，执行退款。",
    }

    evidence, _ = generator.assemble_evidence([hit], [_citation(1)])

    assert "[父级上下文]" in evidence[0]["content"]
    assert "完整退款流程" in evidence[0]["content"]


@pytest.mark.asyncio
async def test_answer_generator_respects_insufficient_evidence_decision():
    async def llm_call(prompt):
        return """{
          "answer": "",
          "used_citations": [],
          "sufficient_evidence": false
        }"""

    result = await RAGAnswerGenerator(llm_call).generate(
        "能否修改收货地址",
        "修改收货地址",
        [_hit("chunk-1", "订单支付后可以查询物流。")],
        [_citation(1)],
    )

    assert result.generated is True
    assert result.sufficient_evidence is False
    assert result.fallback_reason is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer,used_citations",
    [
        ("退款需要十五个工作日。[1]", ["[1]"]),
        ("退款将在七天内到账。[9]", ["[9]"]),
        ("退款将在七天内到账。", []),
    ],
)
async def test_answer_generator_rejects_unsupported_facts_and_citations(
    answer,
    used_citations,
):
    async def llm_call(prompt):
        return json.dumps({
            "answer": answer,
            "used_citations": used_citations,
            "sufficient_evidence": True,
        }, ensure_ascii=False)

    result = await RAGAnswerGenerator(llm_call).generate(
        "退款多久到账",
        "退款到账时间",
        [_hit("chunk-1", "退款将在七天内到账。")],
        [_citation(1)],
    )

    assert result.generated is False
    assert result.sufficient_evidence is False
    assert result.fallback_reason is not None
