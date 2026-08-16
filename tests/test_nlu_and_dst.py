"""Tests for Task 2: structured NLU, DialogueStateTracker, and StateStore.

No real LLM, Redis, or network required.
"""
import json
from types import SimpleNamespace

import pytest

from core.agent_models import (
    INTENT_SCHEMAS,
    ConfirmationStatus,
    DialogueState,
    Observation,
    PendingAction,
    UnderstandingResult,
    UserAct,
)
from core.nlu_fast_track import (
    FastTrackResult,
    build_understanding_from_fast_track,
    detect_intent_from_keywords,
    detect_intents_from_keywords,
    detect_user_act,
    extract_order_id,
    extract_tracking_no,
    fast_track_extract,
)
from core.nlu_llm import (
    NLUServiceUnavailable,
    _build_prompt,
    make_fallback_understanding,
    understand_with_llm,
    _parse_llm_json,
    _validate_llm_output,
)
from core.dialogue_state_tracker import DialogueStateTracker
from core.intent_recognizer import IntentRecognizer
from core.state_store import InMemoryStateStore, RedisStateStore


# ═══════════════════════════════════════════════════════════════════════════════
# 2.1 Fast-track tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractOrderId:
    def test_ord_prefix(self):
        assert extract_order_id("我的订单号 ORD-1001 需要退款") == "ORD-1001"

    def test_order_prefix(self):
        assert extract_order_id("订单号 ORDER-2002") == "ORDER-2002"

    def test_chinese_prefix_digits(self):
        assert extract_order_id("订单号：123456789") == "123456789"

    def test_alphanumeric_code(self):
        result = extract_order_id("查一下 ABC123456789 的状态")
        assert result is not None
        assert len(result) >= 8

    def test_no_match(self):
        assert extract_order_id("你好，我想退款") is None

    def test_correction_marks_order_id(self):
        result = fast_track_extract("不对，订单号是 ORD-1002")
        assert result.corrected_slots == ["order_id"]


class TestExtractTrackingNo:
    def test_sf_tracking(self):
        assert extract_tracking_no("物流单号 SF1234567890") == "SF1234567890"

    def test_jd_tracking(self):
        assert extract_tracking_no("JD009876543210") == "JD009876543210"

    def test_yt_tracking(self):
        assert extract_tracking_no("YT123456789012") == "YT123456789012"

    def test_chinese_prefix(self):
        assert extract_tracking_no("运单号 12345678901234") == "12345678901234"

    def test_no_match(self):
        assert extract_tracking_no("你好") is None


class TestDetectUserAct:
    def test_confirm(self):
        assert detect_user_act("确认", confirmation_pending=True) == UserAct.CONFIRM
        assert detect_user_act("是的", confirmation_pending=True) == UserAct.CONFIRM
        assert detect_user_act("好的", confirmation_pending=True) == UserAct.CONFIRM

    def test_reject(self):
        assert detect_user_act("取消", confirmation_pending=True) == UserAct.REJECT
        assert detect_user_act("不要", confirmation_pending=True) == UserAct.REJECT
        assert detect_user_act("不办了", confirmation_pending=True) == UserAct.REJECT

    @pytest.mark.parametrize("text", ["不可以", "不行", "不要确认", "不能提交"])
    def test_negative_confirmation_is_rejected(self, text):
        assert detect_user_act(text, confirmation_pending=True) == UserAct.REJECT

    @pytest.mark.parametrize(
        "text",
        ["yes", "ok", "confirm", "no", "cancel", "reject"],
    )
    def test_english_user_acts_are_ignored(self, text):
        assert detect_user_act(text, confirmation_pending=True) is None

    def test_explicit_goal_switch(self):
        assert detect_user_act("算了不退了，我想查物流") == UserAct.SWITCH

    def test_business_intent_takes_priority_over_pending_rejection(self):
        assert detect_user_act(
            "取消订单",
            confirmation_pending=True,
        ) is None

    def test_confirmation_words_are_ignored_without_pending_action(self):
        assert detect_user_act("确认", confirmation_pending=False) is None
        assert detect_user_act("不办了", confirmation_pending=False) is None

    def test_neither(self):
        assert detect_user_act("我要退款") is None


class TestDetectIntentFromKeywords:
    def test_refund(self):
        assert detect_intent_from_keywords("我要退款") == "refund_request"

    def test_logistics(self):
        assert detect_intent_from_keywords("查物流") == "logistics_query"

    def test_delivery_policy_is_a_knowledge_query(self):
        assert detect_intent_from_keywords("配送一般几天") == "query"
        assert detect_intents_from_keywords("配送一般几天") == ["query"]
        assert detect_intent_from_keywords("一般配送需要几天") == "query"
        assert detect_intent_from_keywords("物流通常多久能到") == "query"

    def test_no_match(self):
        assert detect_intent_from_keywords("你好") is None

    def test_compound_order_and_logistics_intents(self):
        intents = detect_intents_from_keywords("查 ORD-1002 的订单和物流")

        assert intents == ["logistics_query", "order_query"]

    def test_handoff_intents_have_deterministic_routes(self):
        assert detect_intent_from_keywords("我要投诉") == "complaint"
        assert detect_intent_from_keywords("转人工客服") == "escalation"

    @pytest.mark.parametrize(
        "text",
        [
            "complain about service",
            "refund status",
            "return value",
            "cancel subscription",
            "track performance",
            "order by time",
        ],
    )
    def test_english_intent_keywords_are_ignored(self, text):
        assert detect_intents_from_keywords(text) == []

    @pytest.mark.parametrize(
        "text",
        [
            "退款率报表",
            "退货率分析",
            "物流行业报告",
            "订单数据统计",
            "投诉率趋势",
        ],
    )
    def test_analytics_terms_do_not_trigger_customer_service_intents(self, text):
        assert detect_intents_from_keywords(text) == []

    @pytest.mark.parametrize(
        ("text", "intent"),
        [
            ("退款什么时候到账", "refund_request"),
            ("我要申请退货", "return_request"),
            ("我的订单状态是什么", "order_query"),
            ("帮我查一下物流", "logistics_query"),
        ],
    )
    def test_chinese_customer_service_phrases_remain_supported(
        self,
        text,
        intent,
    ):
        assert detect_intent_from_keywords(text) == intent


class TestFastTrackExtract:
    def test_order_id_and_intent(self):
        ft = fast_track_extract("我要退款，订单号 ORD-1001")
        assert ft.order_id == "ORD-1001"
        assert ft.intent == "refund_request"
        assert ft.hit is True
        assert ft.slots["order_id"] == "ORD-1001"

    def test_tracking_and_intent(self):
        ft = fast_track_extract("查快递 SF1234567890")
        assert ft.tracking_no == "SF1234567890"
        assert ft.intent == "logistics_query"

    def test_confirm_word(self):
        ft = fast_track_extract("确认", confirmation_pending=True)
        assert ft.user_act == UserAct.CONFIRM
        assert ft.hit is True

    def test_slot_correction_is_not_treated_as_rejection(self):
        ft = fast_track_extract(
            "不对，订单号是 ORD-1002",
            confirmation_pending=True,
        )
        assert ft.corrected_slots == ["order_id"]
        assert ft.user_act == UserAct.INFORM

    def test_no_hit(self):
        ft = fast_track_extract("你好")
        assert ft.hit is False

    def test_compound_fast_track_keeps_all_intents(self):
        ft = fast_track_extract("查 ORD-1002 的订单和物流")
        result = build_understanding_from_fast_track(
            ft,
            "查 ORD-1002 的订单和物流",
        )

        assert result.intents == ["logistics_query", "order_query"]
        assert result.primary_intent == "logistics_query"


class TestBuildUnderstandingFromFastTrack:
    def test_full_fast_track(self):
        ft = FastTrackResult(order_id="ORD-1001", intent="refund_request")
        result = build_understanding_from_fast_track(ft, "我要退款，订单号 ORD-1001")
        assert result is not None
        assert result.primary_intent == "refund_request"
        assert result.extracted_slots["order_id"] == "ORD-1001"
        assert result.confidence == 1.0

    def test_no_hit_returns_none(self):
        ft = FastTrackResult()
        assert build_understanding_from_fast_track(ft, "你好") is None

    def test_order_id_infers_order_query(self):
        ft = FastTrackResult(order_id="ORD-1001")
        result = build_understanding_from_fast_track(ft, "ORD-1001")
        assert result is not None
        assert result.primary_intent == "order_query"

    def test_broad_refund_keyword_does_not_short_circuit_llm(self):
        ft = fast_track_extract("退款什么时候到账")
        result = build_understanding_from_fast_track(
            ft,
            "退款什么时候到账",
        )

        assert result.primary_intent == "refund_request"
        assert result.confidence < 0.9

    def test_tracking_number_infers_logistics_query(self):
        ft = FastTrackResult(tracking_no="SF1234567890")
        result = build_understanding_from_fast_track(ft, "SF1234567890")

        assert result.primary_intent == "logistics_query"
        assert result.confidence == 0.9


class TestStructuredRecognizerFastTrack:
    class FakeMessages:
        def __init__(self, payload):
            self.payload = payload
            self.calls = 0

        async def create(self, **kwargs):
            del kwargs
            self.calls += 1
            return SimpleNamespace(content=[json.dumps(self.payload)])

    @classmethod
    def recognizer_with_response(cls, payload):
        recognizer = IntentRecognizer.__new__(IntentRecognizer)
        recognizer.model = "test-model"
        messages = cls.FakeMessages(payload)
        recognizer.client = SimpleNamespace(messages=messages)
        recognizer._slot_validator = None
        return recognizer, messages

    def test_slot_signals_only_support_chinese_labels(self):
        assert IntentRecognizer._has_slot_signal("订单号是 ORD-1001", "order_id")
        assert IntentRecognizer._has_slot_signal(
            "物流单号是 SF1234567890",
            "tracking_no",
        )
        assert not IntentRecognizer._has_slot_signal("order id ORD-1001", "order_id")
        assert not IntentRecognizer._has_slot_signal(
            "tracking number SF1234567890",
            "tracking_no",
        )

    def test_legacy_llm_recognizer_remains_a_class_method(self):
        assert callable(IntentRecognizer._llm_recognize)

    @pytest.mark.asyncio
    async def test_slot_only_turn_inherits_active_intent_without_llm(self):
        recognizer = IntentRecognizer.__new__(IntentRecognizer)
        result = await recognizer.recognize_structured(
            "ORD-1001",
            current_state={"active_intent": "refund_request"},
        )
        assert result.primary_intent == "refund_request"
        assert result.extracted_slots == {"order_id": "ORD-1001"}

    @pytest.mark.asyncio
    async def test_confirm_turn_inherits_active_intent_without_llm(self):
        recognizer = IntentRecognizer.__new__(IntentRecognizer)
        result = await recognizer.recognize_structured(
            "确认",
            current_state={
                "active_intent": "refund_request",
                "confirmation_status": "pending",
            },
        )
        assert result.primary_intent == "refund_request"
        assert result.user_act == UserAct.CONFIRM

    @pytest.mark.asyncio
    async def test_correction_inherits_active_intent_without_llm(self):
        recognizer = IntentRecognizer.__new__(IntentRecognizer)
        result = await recognizer.recognize_structured(
            "不对，订单号是 ORD-1002",
            current_state={"active_intent": "refund_request"},
        )
        assert result.primary_intent == "refund_request"
        assert result.corrected_slots == ["order_id"]
        assert result.user_act == UserAct.INFORM

    @pytest.mark.asyncio
    async def test_explicit_new_goal_becomes_switch_without_switch_words(self):
        recognizer = IntentRecognizer.__new__(IntentRecognizer)
        result = await recognizer.recognize_structured(
            "查物流",
            current_state={"active_intent": "refund_request"},
        )

        assert result.primary_intent == "logistics_query"
        assert result.user_act == UserAct.SWITCH

    @pytest.mark.asyncio
    async def test_missing_slot_without_slot_signal_skips_llm(self):
        recognizer, messages = self.recognizer_with_response({
            "intent": "refund_request",
            "confidence": 0.95,
            "slots": {"order_id": "ORD-9999"},
        })

        result = await recognizer.recognize_structured("我要退款")

        assert messages.calls == 0
        assert result.primary_intent == "refund_request"
        assert result.extracted_slots == {}

    @pytest.mark.asyncio
    async def test_slot_signal_uses_llm_and_accepts_grounded_business_value(self):
        recognizer, messages = self.recognizer_with_response({
            "intent": "refund_request",
            "confidence": 0.95,
            "slots": {"order_id": "ab-123-xyz"},
            "user_act": "inform",
        })
        validated = []

        async def validate(slot_name, value):
            validated.append((slot_name, value))
            return True

        recognizer.set_slot_validator(validate)
        result = await recognizer.recognize_structured(
            "我要退款，订单编号是 ab-123-xyz",
        )

        assert messages.calls == 1
        assert validated == [("order_id", "AB-123-XYZ")]
        assert result.primary_intent == "refund_request"
        assert result.extracted_slots == {"order_id": "AB-123-XYZ"}

    @pytest.mark.asyncio
    async def test_llm_slot_not_present_in_message_is_rejected(self):
        recognizer, messages = self.recognizer_with_response({
            "intent": "refund_request",
            "confidence": 0.95,
            "slots": {"order_id": "ORD-9999"},
            "user_act": "inform",
        })
        validator_calls = []

        async def validate(slot_name, value):
            validator_calls.append((slot_name, value))
            return True

        recognizer.set_slot_validator(validate)
        result = await recognizer.recognize_structured(
            "我要退款，订单编号在截图里",
        )

        assert messages.calls == 1
        assert validator_calls == []
        assert result.extracted_slots == {}

    @pytest.mark.asyncio
    async def test_grounded_llm_slot_rejected_when_business_validation_fails(self):
        recognizer, messages = self.recognizer_with_response({
            "intent": "logistics_query",
            "confidence": 0.95,
            "slots": {"tracking_no": "carrier-12345"},
            "user_act": "inform",
        })

        async def validate(slot_name, value):
            assert (slot_name, value) == (
                "tracking_no",
                "CARRIER-12345",
            )
            return False

        recognizer.set_slot_validator(validate)
        result = await recognizer.recognize_structured(
            "帮我查物流，运单号 carrier-12345",
        )

        assert messages.calls == 1
        assert result.primary_intent == "logistics_query"
        assert result.extracted_slots == {}


# ═══════════════════════════════════════════════════════════════════════════════
# 2.2 LLM structured call tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseLlmJson:
    def test_valid_json(self):
        data = _parse_llm_json('{"intent":"refund_request","confidence":0.9}')
        assert data == {"intent": "refund_request", "confidence": 0.9}

    def test_json_with_surrounding_text(self):
        data = _parse_llm_json('Here is the result:\n{"intent":"query","confidence":0.8}\nDone')
        assert data is None

    def test_markdown_fenced(self):
        data = _parse_llm_json('```json\n{"intent":"other","confidence":0.5}\n```')
        assert data is None

    def test_invalid_json(self):
        assert _parse_llm_json("not json at all") is None

    def test_empty_string(self):
        assert _parse_llm_json("") is None


class TestValidateLlmOutput:
    def test_valid_output(self):
        data = {
            "intent": "refund_request",
            "confidence": 0.95,
            "slots": {"order_id": "ORD-1001"},
            "user_act": "inform",
            "corrected_slots": [],
        }
        result = _validate_llm_output(data)
        assert result is not None
        assert result.primary_intent == "refund_request"
        assert result.extracted_slots["order_id"] == "ORD-1001"

    def test_unknown_intent_fallback(self):
        data = {"intent": "unknown_intent", "confidence": 0.9, "slots": {}}
        result = _validate_llm_output(data)
        assert result is not None
        assert result.primary_intent == "other"
        assert result.confidence <= 0.5

    def test_invalid_user_act(self):
        data = {"intent": "query", "confidence": 0.8, "slots": {}, "user_act": "bad_act"}
        result = _validate_llm_output(data)
        assert result is not None
        assert result.user_act == UserAct.INFORM

    def test_missing_intent(self):
        data = {"confidence": 0.8, "slots": {}}
        assert _validate_llm_output(data) is None

    def test_valid_output_keeps_multiple_intents(self):
        result = _validate_llm_output({
            "intent": "order_query",
            "intents": ["order_query", "logistics_query", "order_query"],
            "confidence": 0.9,
            "slots": {"order_id": "ORD-1001"},
        })

        assert result.intents == ["order_query", "logistics_query"]


class TestMakeFallbackUnderstanding:
    def test_pure_other(self):
        result = make_fallback_understanding("你好")
        assert result.primary_intent == "other"
        assert result.confidence == 0.3

    def test_with_fast_track_slots(self):
        result = make_fallback_understanding("订单号 ORD-1001")
        assert result.primary_intent == "order_query"
        assert result.extracted_slots.get("order_id") == "ORD-1001"
        assert result.confidence == 0.6

    def test_with_fast_track_correction(self):
        result = make_fallback_understanding("不对，订单号是 ORD-1002")
        assert result.corrected_slots == ["order_id"]


class TestUnderstandWithLlm:
    def test_prompt_contains_bounded_recent_history(self):
        prompt = _build_prompt(
            "它到哪里了",
            {"active_intent": "logistics_query"},
            [
                {"role": "user", "content": "查询 ORD-1001"},
                {"role": "assistant", "content": "订单已经发货"},
            ],
        )

        assert "recent_dialogue" in prompt
        assert "查询 ORD-1001" in prompt
        assert "它到哪里了" in prompt

    @pytest.mark.asyncio
    async def test_valid_llm_response(self):
        async def mock_llm(prompt: str) -> str:
            return json.dumps({
                "intent": "refund_request",
                "confidence": 0.92,
                "slots": {"order_id": "ORD-1001"},
                "user_act": "inform",
                "corrected_slots": [],
            })

        result = await understand_with_llm("我要退款，订单号 ORD-1001", mock_llm)
        assert result.primary_intent == "refund_request"
        assert result.extracted_slots["order_id"] == "ORD-1001"
        assert result.confidence == 0.92

    @pytest.mark.asyncio
    async def test_invalid_json_degrades(self):
        async def mock_llm(prompt: str) -> str:
            return "I cannot understand the request"

        result = await understand_with_llm("你好", mock_llm)
        assert result.primary_intent == "other"
        assert result.confidence <= 0.5

    @pytest.mark.asyncio
    async def test_exception_degrades(self):
        """LLM backend failure must surface as NLUServiceUnavailable so the
        runtime can return a service-outage message instead of masking the
        outage as 'user intent unclear'."""
        async def mock_llm(prompt: str) -> str:
            raise RuntimeError("LLM unavailable")

        with pytest.raises(NLUServiceUnavailable):
            await understand_with_llm("测试", mock_llm)

    @pytest.mark.asyncio
    async def test_empty_output_degrades(self):
        async def mock_llm(prompt: str) -> str:
            return ""

        result = await understand_with_llm("测试", mock_llm)
        assert result.primary_intent == "other"

    @pytest.mark.asyncio
    async def test_correction_detected(self):
        async def mock_llm(prompt: str) -> str:
            return json.dumps({
                "intent": "refund_request",
                "confidence": 0.9,
                "slots": {"order_id": "ORD-1002"},
                "user_act": "inform",
                "corrected_slots": ["order_id"],
            })

        result = await understand_with_llm("不对，订单号是 ORD-1002", mock_llm)
        assert "order_id" in result.corrected_slots
        assert result.extracted_slots["order_id"] == "ORD-1002"

    @pytest.mark.asyncio
    async def test_llm_confirmation_requires_pending_action(self):
        async def mock_llm(prompt: str) -> str:
            return json.dumps({
                "intent": "other",
                "confidence": 0.9,
                "slots": {},
                "user_act": "confirm",
                "corrected_slots": [],
            })

        without_pending = await understand_with_llm("确认", mock_llm)
        with_pending = await understand_with_llm(
            "确认",
            mock_llm,
            {
                "active_intent": "refund_request",
                "confirmation_status": "pending",
            },
        )

        assert without_pending.user_act == UserAct.INFORM
        assert with_pending.user_act == UserAct.CONFIRM

    @pytest.mark.asyncio
    async def test_llm_correction_takes_priority_over_rejection(self):
        async def mock_llm(prompt: str) -> str:
            return json.dumps({
                "intent": "refund_request",
                "confidence": 0.9,
                "slots": {"order_id": "ORD-1002"},
                "user_act": "reject",
                "corrected_slots": ["order_id"],
            })

        result = await understand_with_llm(
            "不对，订单号是 ORD-1002",
            mock_llm,
            {
                "active_intent": "refund_request",
                "confirmation_status": "pending",
            },
        )

        assert result.user_act == UserAct.INFORM


# ═══════════════════════════════════════════════════════════════════════════════
# 2.3 DialogueStateTracker Reducer tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDialogueStateTracker:
    def setup_method(self):
        self.tracker = DialogueStateTracker()

    def _understanding(
        self,
        intent: str = "refund_request",
        slots: dict = None,
        user_act: UserAct = UserAct.INFORM,
        corrected: list = None,
    ) -> UnderstandingResult:
        return UnderstandingResult(
            intents=[intent],
            primary_intent=intent,
            confidence=0.9,
            extracted_slots=slots or {},
            corrected_slots=corrected or [],
            user_act=user_act,
        )

    def test_initial_state_gets_intent_and_slots(self):
        state = DialogueState()
        u = self._understanding("refund_request", {"order_id": "ORD-1001"})
        new_state = self.tracker.update(state, u)

        assert new_state.active_intent == "refund_request"
        assert new_state.slots["order_id"] == "ORD-1001"
        assert new_state.state_version == 1
        # order_id is required for refund_request, so missing_slots should be empty
        assert new_state.missing_slots == []
        assert new_state.required_slots == ["order_id"]

    def test_missing_slot_detected(self):
        state = DialogueState()
        u = self._understanding("refund_request", {})  # no order_id
        new_state = self.tracker.update(state, u)

        assert new_state.active_intent == "refund_request"
        assert "order_id" in new_state.missing_slots
        assert "order_id" in new_state.required_slots

    def test_logistics_accepts_order_or_tracking_number(self):
        order_state = self.tracker.update(
            DialogueState(),
            self._understanding(
                "logistics_query",
                {"order_id": "ORD-1001"},
            ),
        )
        tracking_state = self.tracker.update(
            DialogueState(),
            self._understanding(
                "logistics_query",
                {"tracking_no": "SF1234567890"},
            ),
        )

        assert order_state.missing_slots == []
        assert tracking_state.missing_slots == []

    def test_slot_inheritance_across_turns(self):
        """Turn 1: user says '我要退款' (no order_id).
           Turn 2: user provides order_id -> inherited refund intent, slot filled."""
        state = DialogueState()
        u1 = self._understanding("refund_request", {})
        state = self.tracker.update(state, u1)
        assert "order_id" in state.missing_slots

        u2 = UnderstandingResult(
            intents=["order_query", "refund_request"],
            primary_intent="order_query",  # fast-track may infer order_query
            confidence=0.9,
            extracted_slots={"order_id": "ORD-1001"},
            user_act=UserAct.INFORM,
        )
        state = self.tracker.update(state, u2)

        # The refund_request intent from turn 1 should persist because
        # this is not an explicit switch; it's just a slot-filling turn.
        assert state.active_intent == "refund_request"
        assert state.slots["order_id"] == "ORD-1001"
        assert state.missing_slots == []

    def test_cross_turn_slot_filling_preserves_intent(self):
        """User says '我要退款', then next turn just provides order_id.
        The refund intent should be preserved."""
        state = DialogueState()
        u1 = self._understanding("refund_request", {})
        state = self.tracker.update(state, u1)

        u2 = UnderstandingResult(
            intents=["refund_request"],
            primary_intent="refund_request",
            confidence=0.9,
            extracted_slots={"order_id": "ORD-1001"},
            user_act=UserAct.INFORM,
        )
        state = self.tracker.update(state, u2)

        assert state.active_intent == "refund_request"
        assert state.slots["order_id"] == "ORD-1001"
        assert state.missing_slots == []

    def test_slot_correction(self):
        state = DialogueState()
        u1 = self._understanding("refund_request", {"order_id": "ORD-1001"})
        state = self.tracker.update(state, u1)
        assert state.slots["order_id"] == "ORD-1001"

        u2 = self._understanding(
            "refund_request",
            {"order_id": "ORD-1002"},
            corrected=["order_id"],
        )
        state = self.tracker.update(state, u2)
        assert state.slots["order_id"] == "ORD-1002"
        assert state.state_version == 2

    def test_intent_switch_clears_pending_and_non_reuse_slots(self):
        state = DialogueState()
        u1 = self._understanding("refund_request", {"order_id": "ORD-1001"})
        state = self.tracker.update(state, u1)
        state = DialogueStateTracker.set_pending_action(
            state,
            PendingAction(tool_name="create_refund", arguments={"order_id": "ORD-1001"}),
        )
        assert state.pending_action is not None

        u2 = UnderstandingResult(
            intents=["logistics_query"],
            primary_intent="logistics_query",
            confidence=0.9,
            extracted_slots={},
            user_act=UserAct.SWITCH,
        )
        state = self.tracker.update(state, u2)

        assert state.active_intent == "logistics_query"
        assert state.pending_action is None
        assert state.confirmation_status == ConfirmationStatus.NOT_REQUIRED
        # order_id is cross-intent reusable
        assert state.slots["order_id"] == "ORD-1001"

    def test_intent_switch_non_reuse_slots_cleared(self):
        state = DialogueState()
        u1 = self._understanding("refund_request", {"order_id": "ORD-1001", "refund_reason": "defective"})
        state = self.tracker.update(state, u1)
        assert state.slots.get("refund_reason") == "defective"

        u2 = UnderstandingResult(
            intents=["logistics_query"],
            primary_intent="logistics_query",
            confidence=0.9,
            extracted_slots={},
            user_act=UserAct.SWITCH,
        )
        state = self.tracker.update(state, u2)

        assert state.active_intent == "logistics_query"
        # order_id is cross-intent reusable, refund_reason is not
        assert state.slots.get("order_id") == "ORD-1001"
        assert "refund_reason" not in state.slots

    def test_confirmation_advances_state(self):
        state = DialogueState()
        state = DialogueStateTracker.set_pending_action(
            state,
            PendingAction(tool_name="create_refund", arguments={"order_id": "ORD-1001"}),
        )
        assert state.confirmation_status == ConfirmationStatus.PENDING

        u = self._understanding("refund_request", {}, user_act=UserAct.CONFIRM)
        state = self.tracker.update(state, u)
        assert state.confirmation_status == ConfirmationStatus.CONFIRMED

    def test_rejection_clears_pending(self):
        state = DialogueState()
        state = DialogueStateTracker.set_pending_action(
            state,
            PendingAction(tool_name="create_refund", arguments={"order_id": "ORD-1001"}),
        )

        u = self._understanding("refund_request", {}, user_act=UserAct.REJECT)
        state = self.tracker.update(state, u)
        assert state.confirmation_status == ConfirmationStatus.REJECTED
        assert state.pending_action is None

    def test_observation_merges_tool_results(self):
        state = DialogueState()
        u = self._understanding("order_query", {"order_id": "ORD-1001"})
        state = self.tracker.update(state, u)
        assert "order_status" not in state.slots

        obs = Observation(
            source="tool",
            name="query_order",
            success=True,
            data={"order_status": "shipped"},
        )
        state = self.tracker.update(state, self._understanding("order_query"), [obs])
        assert state.slots["order_status"] == "shipped"

    def test_observation_does_not_override_user_slot(self):
        state = DialogueState()
        u1 = self._understanding("order_query", {"order_id": "ORD-1001"})
        state = self.tracker.update(state, u1)

        obs = Observation(
            source="tool",
            name="query_order",
            success=True,
            data={"order_id": "DIFFERENT-ID"},
        )
        state = self.tracker.update(state, self._understanding("order_query"), [obs])
        # User-provided value should not be overwritten by tool result
        assert state.slots["order_id"] == "ORD-1001"

    def test_mark_goal_completed(self):
        state = DialogueState()
        state = DialogueStateTracker.mark_goal_completed(state, "order_query")
        assert state.completed_goals == ["order_query"]
        assert state.state_version == 1

        state = DialogueStateTracker.mark_goal_completed(state, "order_query")
        assert state.completed_goals == ["order_query"]  # no duplicates

    def test_version_increments_each_update(self):
        state = DialogueState()
        assert state.state_version == 0

        u = self._understanding("order_query", {"order_id": "ORD-1001"})
        state = self.tracker.update(state, u)
        assert state.state_version == 1

        state = self.tracker.update(state, self._understanding("order_query"))
        assert state.state_version == 2

    def test_last_agent_set_from_schema(self):
        state = DialogueState()
        u = self._understanding("refund_request", {"order_id": "ORD-1001"})
        state = self.tracker.update(state, u)
        assert state.last_agent == "after_sales"


# ═══════════════════════════════════════════════════════════════════════════════
# 2.4 StateStore tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestInMemoryStateStore:
    @pytest.mark.asyncio
    async def test_save_and_load(self):
        store = InMemoryStateStore()
        state = DialogueState(
            active_intent="refund_request",
            slots={"order_id": "ORD-1001"},
            required_slots=["order_id"],
            state_version=3,
        )
        await store.save("u1", "c1", state)

        loaded = await store.load("u1", "c1")
        assert loaded is not None
        assert loaded.active_intent == "refund_request"
        assert loaded.slots["order_id"] == "ORD-1001"
        assert loaded.state_version == 3

    @pytest.mark.asyncio
    async def test_load_missing_returns_none(self):
        store = InMemoryStateStore()
        result = await store.load("missing", "missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self):
        store = InMemoryStateStore()
        state = DialogueState(active_intent="query")
        await store.save("u1", "c1", state)
        await store.delete("u1", "c1")
        assert await store.load("u1", "c1") is None

    @pytest.mark.asyncio
    async def test_overwrite_on_save(self):
        store = InMemoryStateStore()
        state1 = DialogueState(active_intent="query", state_version=1)
        await store.save("u1", "c1", state1)

        state2 = DialogueState(active_intent="refund_request", state_version=2)
        await store.save("u1", "c1", state2)

        loaded = await store.load("u1", "c1")
        assert loaded.active_intent == "refund_request"
        assert loaded.state_version == 2

    @pytest.mark.asyncio
    async def test_independent_conversations(self):
        store = InMemoryStateStore()
        await store.save("u1", "c1", DialogueState(active_intent="query"))
        await store.save("u1", "c2", DialogueState(active_intent="refund_request"))

        c1 = await store.load("u1", "c1")
        c2 = await store.load("u1", "c2")
        assert c1.active_intent == "query"
        assert c2.active_intent == "refund_request"

    @pytest.mark.asyncio
    async def test_json_round_trip(self):
        """Ensure state with complex fields survives serialization."""
        store = InMemoryStateStore()
        state = DialogueState(
            active_intent="refund_request",
            slots={"order_id": "ORD-1001"},
            required_slots=["order_id"],
            pending_action=PendingAction(
                tool_name="create_refund",
                arguments={"order_id": "ORD-1001"},
            ),
            confirmation_status=ConfirmationStatus.PENDING,
            last_agent="after_sales",
            state_version=5,
        )
        await store.save("u1", "c1", state)
        loaded = await store.load("u1", "c1")

        assert loaded == state
        assert loaded.pending_action is not None
        assert loaded.pending_action.tool_name == "create_refund"


class _FakeRedis:
    def __init__(self):
        self.data = {}
        self.ttls = {}

    def get(self, key):
        return self.data.get(key)

    def setex(self, key, ttl, value):
        self.data[key] = value
        self.ttls[key] = ttl

    def delete(self, key):
        self.data.pop(key, None)


class TestRedisStateStore:
    @pytest.mark.asyncio
    async def test_save_load_and_delete(self):
        redis = _FakeRedis()
        store = RedisStateStore(redis, ttl=60)
        state = DialogueState(
            active_intent="refund_request",
            required_slots=["order_id"],
            missing_slots=["order_id"],
            state_version=2,
        )

        await store.save("u1", "c1", state)
        assert redis.ttls["dst:u1:c1"] == 60
        assert await store.load("u1", "c1") == state

        await store.delete("u1", "c1")
        assert await store.load("u1", "c1") is None

    @pytest.mark.asyncio
    async def test_load_accepts_bytes(self):
        redis = _FakeRedis()
        state = DialogueState(active_intent="order_query")
        redis.data["dst:u1:c1"] = state.model_dump_json().encode("utf-8")

        loaded = await RedisStateStore(redis).load("u1", "c1")
        assert loaded == state


# ═══════════════════════════════════════════════════════════════════════════════
# 2.5 Multi-turn integration tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiTurnIntegration:
    """End-to-end multi-turn scenarios combining fast-track, LLM fallback,
    DST tracker, and state store."""

    def setup_method(self):
        self.tracker = DialogueStateTracker()
        self.store = InMemoryStateStore()

    @pytest.mark.asyncio
    async def test_refund_full_flow(self):
        """User: 我要退款 -> missing order_id -> User: ORD-1001 -> confirmed."""
        # Turn 1: fast-track detects intent but no order_id
        ft1 = fast_track_extract("我要退款")
        u1 = build_understanding_from_fast_track(ft1, "我要退款")
        if u1 is None:
            # Fall back to manual construction
            u1 = UnderstandingResult(
                intents=["refund_request"],
                primary_intent="refund_request",
                confidence=0.9,
                extracted_slots={},
                user_act=UserAct.INFORM,
                route_to="after_sales",
            )

        state = DialogueState()
        state = self.tracker.update(state, u1)
        await self.store.save("u1", "c1", state)

        assert state.active_intent == "refund_request"
        assert "order_id" in state.missing_slots

        # Turn 2: user provides order_id
        ft2 = fast_track_extract("ORD-1001")
        u2 = build_understanding_from_fast_track(ft2, "ORD-1001")
        if u2 is None:
            u2 = UnderstandingResult(
                intents=["order_query"],
                primary_intent="order_query",
                confidence=0.9,
                extracted_slots={"order_id": "ORD-1001"},
                user_act=UserAct.INFORM,
                route_to="order",
            )

        state = await self.store.load("u1", "c1")
        state = self.tracker.update(state, u2)
        await self.store.save("u1", "c1", state)

        assert state.slots["order_id"] == "ORD-1001"

    @pytest.mark.asyncio
    async def test_correction_flow(self):
        """User provides wrong order_id, then corrects it."""
        state = DialogueState()
        u1 = UnderstandingResult(
            intents=["refund_request"],
            primary_intent="refund_request",
            confidence=0.9,
            extracted_slots={"order_id": "ORD-1001"},
            user_act=UserAct.INFORM,
        )
        state = self.tracker.update(state, u1)
        assert state.slots["order_id"] == "ORD-1001"

        u2 = UnderstandingResult(
            intents=["refund_request"],
            primary_intent="refund_request",
            confidence=0.9,
            extracted_slots={"order_id": "ORD-1002"},
            corrected_slots=["order_id"],
            user_act=UserAct.INFORM,
        )
        state = self.tracker.update(state, u2)
        assert state.slots["order_id"] == "ORD-1002"
        assert state.state_version == 2

    @pytest.mark.asyncio
    async def test_intent_switch_flow(self):
        """User starts refund, then switches to logistics query."""
        state = DialogueState()
        u1 = UnderstandingResult(
            intents=["refund_request"],
            primary_intent="refund_request",
            confidence=0.9,
            extracted_slots={"order_id": "ORD-1001"},
            user_act=UserAct.INFORM,
        )
        state = self.tracker.update(state, u1)
        state = DialogueStateTracker.set_pending_action(
            state,
            PendingAction(tool_name="create_refund", arguments={"order_id": "ORD-1001"}),
        )
        await self.store.save("u1", "c1", state)

        # User switches intent
        u2 = UnderstandingResult(
            intents=["logistics_query"],
            primary_intent="logistics_query",
            confidence=0.9,
            extracted_slots={},
            user_act=UserAct.SWITCH,
        )
        state = await self.store.load("u1", "c1")
        state = self.tracker.update(state, u2)

        assert state.active_intent == "logistics_query"
        assert state.pending_action is None
        assert state.confirmation_status == ConfirmationStatus.NOT_REQUIRED
        # order_id survives the switch
        assert state.slots.get("order_id") == "ORD-1001"

    @pytest.mark.asyncio
    async def test_refund_to_logistics_clears_refund_state(self):
        state = DialogueState(
            active_intent="refund_request",
            slots={
                "order_id": "ORD-1001",
                "refund_reason": "不想要了",
                "refund_amount": 99,
            },
            required_slots=["order_id"],
            pending_action=PendingAction(
                tool_name="create_refund",
                arguments={"order_id": "ORD-1001"},
            ),
            confirmation_status=ConfirmationStatus.PENDING,
        )

        message = "算了不退了，我想查物流"
        understanding = build_understanding_from_fast_track(
            fast_track_extract(message),
            message,
        )
        assert understanding is not None
        assert understanding.user_act == UserAct.SWITCH

        state = self.tracker.update(state, understanding)

        assert state.active_intent == "logistics_query"
        assert state.slots == {"order_id": "ORD-1001"}
        assert state.pending_action is None
        assert state.confirmation_status == ConfirmationStatus.NOT_REQUIRED

    @pytest.mark.asyncio
    async def test_confirm_reject_cycle(self):
        """User confirms, then rejects after reconsideration."""
        state = DialogueState()
        state = DialogueStateTracker.set_pending_action(
            state,
            PendingAction(tool_name="create_refund", arguments={"order_id": "ORD-1001"}),
        )
        assert state.confirmation_status == ConfirmationStatus.PENDING

        # First confirm
        u1 = UnderstandingResult(
            intents=["refund_request"],
            primary_intent="refund_request",
            confidence=0.9,
            user_act=UserAct.CONFIRM,
        )
        state = self.tracker.update(state, u1)
        assert state.confirmation_status == ConfirmationStatus.CONFIRMED
        assert state.active_intent == "refund_request"

        # Now reject (user changed mind after confirmation)
        u2 = UnderstandingResult(
            intents=["refund_request"],
            primary_intent="refund_request",
            confidence=0.9,
            user_act=UserAct.REJECT,
        )
        state = self.tracker.update(state, u2)
        assert state.confirmation_status == ConfirmationStatus.REJECTED
        assert state.pending_action is None
        assert state.active_intent == "refund_request"

    @pytest.mark.asyncio
    async def test_llm_degradation_integrates_with_tracker(self):
        """When LLM returns garbage, the fallback still produces a valid
        UnderstandingResult that the tracker can process."""
        async def bad_llm(prompt):
            return "I don't understand"

        result = await understand_with_llm("你好", bad_llm)
        assert result.primary_intent == "other"

        state = DialogueState()
        state = self.tracker.update(state, result)
        # Should not crash; state should be consistent
        assert state.active_intent == "other"
        assert state.state_version == 1
