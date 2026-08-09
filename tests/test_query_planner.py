import pytest

from rag.query_planner import QueryPlanner


@pytest.mark.asyncio
async def test_query_planner_resolves_reference_and_builds_multi_query():
    async def llm_call(prompt):
        assert "它多久到账" in prompt.user
        assert "不得创造" in prompt.system
        return """{
          "standalone_query": "退款审核通过后多久退回原支付账户",
          "alternative_queries": ["退款到账时间", "退款原路退回周期"],
          "resolved_references": {"它": "退款款项"},
          "confidence": 0.92
        }"""

    planner = QueryPlanner(llm_call, max_queries=3)
    plan = await planner.plan(
        "它多久到账",
        history=[
            {"role": "user", "content": "退款政策是什么"},
            {"role": "assistant", "content": "退款将原路退回。"},
        ],
    )

    assert plan.used_llm is True
    assert plan.resolved_references == {"它": "退款款项"}
    assert plan.standalone_query == "退款审核通过后多久退回原支付账户"
    assert len(plan.retrieval_queries(3)) == 3


@pytest.mark.asyncio
async def test_query_planner_falls_back_when_identifier_is_removed():
    async def llm_call(prompt):
        return """{
          "standalone_query": "登录失败如何处理",
          "alternative_queries": [],
          "resolved_references": {},
          "confidence": 0.9
        }"""

    plan = await QueryPlanner(llm_call).plan("ERROR-401 如何处理")

    assert plan.used_llm is False
    assert plan.standalone_query == "ERROR-401 如何处理"
    assert plan.fallback_reason == "ValueError"


@pytest.mark.asyncio
async def test_query_planner_uses_safe_history_fallback_on_llm_error():
    async def llm_call(prompt):
        raise RuntimeError("unavailable")

    plan = await QueryPlanner(llm_call).plan(
        "这个要多久",
        history=[
            {"role": "user", "content": "退款申请已经通过了"},
            {"role": "assistant", "content": "款项会原路退回。"},
        ],
    )

    assert plan.used_llm is False
    assert "退款申请已经通过了" in plan.standalone_query
    assert "这个要多久" in plan.standalone_query
    assert plan.fallback_reason == "RuntimeError"
