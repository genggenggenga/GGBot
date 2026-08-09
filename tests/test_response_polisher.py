import asyncio
import json
from types import SimpleNamespace

from core.response_polisher import (
    PolishRequest,
    ResponseKind,
    ResponsePolisher,
)


def run(coro):
    return asyncio.run(coro)


def test_polisher_improves_eligible_response_and_preserves_facts():
    prompts = []

    async def llm_call(prompt):
        prompts.append(prompt)
        return json.dumps({
            "response": "订单 ORD-1001 当前已发货，详情请参考退款政策。[1]",
            "changed_meaning": False,
        })

    polisher = ResponsePolisher(llm_call, min_chars=0)
    request = polisher.build_request(
        response="订单 ORD-1001 当前状态：已发货。\n\n退款政策请参考相关说明。[1]",
        response_kind=ResponseKind.RAG,
        observations=[
            SimpleNamespace(data={
                "order_id": "ORD-1001",
                "status": "已发货",
            }),
        ],
        citations=[{"citation_id": "[1]", "source": "policy.md"}],
    )

    outcome = run(polisher.polish(request))

    assert outcome.applied is True
    assert outcome.fallback is False
    assert "ORD-1001" in outcome.response
    assert "已发货" in outcome.response
    assert "[1]" in outcome.response
    assert prompts
    payload = json.loads(prompts[0].user.split("\n", 1)[1])
    assert payload["protected_facts"] == {
        "order_id": "ORD-1001",
        "status": "已发货",
    }


def test_polisher_falls_back_when_citation_or_fact_is_removed():
    responses = iter([
        {"response": "订单正在配送。", "changed_meaning": False},
        {"response": "订单 ORD-1001 当前状态：运输中。", "changed_meaning": False},
    ])

    async def llm_call(prompt):
        del prompt
        return json.dumps(next(responses))

    polisher = ResponsePolisher(llm_call, min_chars=0)
    request = PolishRequest(
        original_response="订单 ORD-1001 当前状态：已发货。[1]",
        response_kind=ResponseKind.RAG,
        protected_facts={"order_id": "ORD-1001", "status": "已发货"},
        required_citations=["[1]"],
    )

    missing_fact = run(polisher.polish(request))
    changed_fact = run(polisher.polish(request))

    assert missing_fact.response == request.original_response
    assert missing_fact.validation_error == "protected_fact_missing"
    assert changed_fact.response == request.original_response
    assert changed_fact.validation_error == "protected_fact_missing"


def test_polisher_falls_back_on_new_number_or_execution_stage():
    responses = iter([
        {"response": "退款政策期限为 14 天。[1]", "changed_meaning": False},
        {"response": "退款申请已提交。[1]", "changed_meaning": False},
    ])

    async def llm_call(prompt):
        del prompt
        return json.dumps(next(responses))

    polisher = ResponsePolisher(llm_call, min_chars=0)
    request = PolishRequest(
        original_response="退款政策期限为七天。[1]",
        response_kind=ResponseKind.RAG,
        required_citations=["[1]"],
    )

    new_number = run(polisher.polish(request))
    changed_stage = run(polisher.polish(request))

    assert new_number.validation_error == "new_factual_literal"
    assert changed_stage.validation_error == "execution_stage_changed"
    assert new_number.response == request.original_response
    assert changed_stage.response == request.original_response


def test_polisher_skips_ineligible_or_short_responses():
    calls = 0

    async def llm_call(prompt):
        nonlocal calls
        del prompt
        calls += 1
        return '{"response":"不应调用","changed_meaning":false}'

    polisher = ResponsePolisher(llm_call, min_chars=20)

    simple = run(polisher.polish(PolishRequest(
        original_response="订单状态已返回。",
        response_kind=ResponseKind.SIMPLE_FACT,
    )))
    short_rag = run(polisher.polish(PolishRequest(
        original_response="政策见[1]",
        response_kind=ResponseKind.RAG,
        required_citations=["[1]"],
    )))

    assert simple.applied is False
    assert short_rag.applied is False
    assert calls == 0


def test_polisher_classifies_rag_and_real_multi_agent_responses():
    order = SimpleNamespace(agent="order")
    logistics = SimpleNamespace(agent="logistics")

    assert ResponsePolisher.classify(
        [order],
        [{"citation_id": "[1]"}],
    ) == ResponseKind.RAG
    assert ResponsePolisher.classify(
        [order, logistics],
        [],
    ) == ResponseKind.MULTI_AGENT
    assert ResponsePolisher.classify(
        [order, order],
        [],
    ) == ResponseKind.SIMPLE_FACT


def test_polisher_falls_back_on_changed_meaning_invalid_output_and_timeout():
    request = PolishRequest(
        original_response="退款政策请参考说明。[1]",
        response_kind=ResponseKind.RAG,
        required_citations=["[1]"],
    )

    async def changed_meaning(prompt):
        del prompt
        return (
            '{"response":"退款政策请参考说明。[1]",'
            '"changed_meaning":true}'
        )

    async def invalid_output(prompt):
        del prompt
        return "not-json"

    async def slow_output(prompt):
        del prompt
        await asyncio.sleep(0.02)
        return (
            '{"response":"退款政策请参考说明。[1]",'
            '"changed_meaning":false}'
        )

    changed = run(ResponsePolisher(
        changed_meaning,
        min_chars=0,
    ).polish(request))
    invalid = run(ResponsePolisher(
        invalid_output,
        min_chars=0,
    ).polish(request))
    timeout = run(ResponsePolisher(
        slow_output,
        min_chars=0,
        timeout_s=0.001,
    ).polish(request))

    assert changed.validation_error == "changed_meaning"
    assert invalid.validation_error == "invalid_output"
    assert timeout.validation_error == "timeout"
    assert changed.response == request.original_response
    assert invalid.response == request.original_response
    assert timeout.response == request.original_response
