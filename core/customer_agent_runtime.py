"""State-driven customer-service runtime used by the /chat endpoint."""
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.domain_agents import (
    KNOWLEDGE_AGENT,
    DomainAgentRuntime,
    Router,
)
from core.agent_models import (
    ConfirmationStatus,
    ExecutionState,
    TurnContext,
    TurnEvent,
    TurnEventType,
    UserAct,
    get_intent_schema,
    get_missing_slots,
)
from core.dialogue_state_tracker import DialogueStateTracker
from core.metrics import record_chat_turn
from core.response_polisher import ResponsePolisher
from core.tool_names import logical_tool_name
from core.trace_store import TraceStore, summarize_observations
from core.turn_engine import TurnEngine


@dataclass
class CustomerTurnResult:
    trace_id: str
    response: str
    intent: str
    agent_type: str
    status: str
    escalated: bool
    latency_ms: float
    knowledge_used: bool = False
    missing_slots: List[str] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RuntimeAgentStats:
    total: int = 0
    success: int = 0
    total_latency_ms: float = 0.0

    def record(self, success: bool, latency_ms: float) -> None:
        self.total += 1
        self.success += int(success)
        self.total_latency_ms += latency_ms

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total else 0.0


class CustomerAgentRuntime:
    """Coordinate NLU, DST, TurnEngine and the domain-agent runtime."""

    _ACTION_INTENTS = {
        "refund_request",
        "return_request",
        "cancel_order",
        "request",
        "complaint",
        "escalation",
    }

    def __init__(
        self,
        recognizer: Any,
        tracker: DialogueStateTracker,
        turn_engine: TurnEngine,
        domain_runtime: DomainAgentRuntime,
        router: Router,
        trace_store: TraceStore,
        response_polisher: Optional[ResponsePolisher] = None,
    ) -> None:
        self._recognizer = recognizer
        self._tracker = tracker
        self._turn_engine = turn_engine
        self._domain_runtime = domain_runtime
        self._router = router
        self._trace_store = trace_store
        self._response_polisher = response_polisher
        self._stats: Dict[str, RuntimeAgentStats] = defaultdict(
            RuntimeAgentStats,
        )

    async def run(
        self,
        user_id: str,
        conv_id: str,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        agent_context: str = "",
    ) -> CustomerTurnResult:
        started = time.monotonic()
        trace_id = str(uuid.uuid4())
        engine = self._turn_engine.fork()
        previous = await engine.load_context(user_id, conv_id)
        understanding = await self._recognizer.recognize_structured(
            message,
            history=history,
            current_state=previous.dialogue_state.model_dump(mode="json"),
        )
        intent_clarification = self._intent_clarification(
            previous.dialogue_state,
            understanding,
        )
        if intent_clarification:
            dialogue_state = previous.dialogue_state
            execution_state = ExecutionState.UNDERSTANDING
        else:
            dialogue_state = self._tracker.update(
                previous.dialogue_state,
                understanding,
            )
            execution_state = self._entry_state(engine, dialogue_state)
        execution = TurnContext(
            user_id=user_id,
            conv_id=conv_id,
            dialogue_state=dialogue_state,
            execution_state=execution_state,
            state_history=[execution_state],
        )
        execution_intents = list(understanding.intents)
        if (
            dialogue_state.active_intent
            and understanding.user_act != UserAct.SWITCH
            and understanding.primary_intent != dialogue_state.active_intent
        ):
            execution_intents = [dialogue_state.active_intent]
        if dialogue_state.queued_goals and not intent_clarification:
            execution_intents = list(dict.fromkeys([
                dialogue_state.active_intent,
                *dialogue_state.queued_goals,
            ]))
        turn_data: Dict[str, Any] = {
            "message": message,
            "agent_context": agent_context,
            "history": history,
            "agent": self._router.route(dialogue_state),
            "intents": execution_intents,
            "results": [],
            "citations": [],
            "intent_clarification": intent_clarification,
            "polish_outcome": None,
        }
        self._register_handlers(engine, turn_data)

        # Trace: understanding phase - only intent/slot summaries, no user text
        self._trace_store.append(trace_id, {
            "event": "understanding",
            "intent": understanding.primary_intent,
            "slot_keys": list(understanding.extracted_slots),
            "user_act": understanding.user_act.value,
            "confidence": understanding.confidence,
            "decision": "clarify" if intent_clarification else "accept",
            "state_version": dialogue_state.state_version,
        })

        completed = await engine.run(execution)

        # Trace: agent results with trimmed observation summaries
        for result in turn_data["results"]:
            result_summary = summarize_observations(result.observations)
            self._trace_store.append(trace_id, {
                "event": "agent_result",
                "agent": result.agent,
                "success": result.success,
                "tools": [obs.name for obs in result.observations],
                "result_summary": result_summary,
            })

        polish_outcome = turn_data["polish_outcome"]
        if polish_outcome is not None:
            self._trace_store.append(trace_id, {
                "event": "response_polish",
                "applied": polish_outcome.applied,
                "fallback": polish_outcome.fallback,
                "response_kind": turn_data["response_kind"].value,
                "validation_error": polish_outcome.validation_error,
                "latency_ms": polish_outcome.latency_ms,
            })

        # Trace: RAG-specific summary if rag_search was used
        rag_obs = [
            obs for obs in completed.observations
            if logical_tool_name(obs.name) == "rag_search" and obs.success
        ]
        if rag_obs:
            self._trace_store.append(trace_id, {
                "event": "rag_retrieval",
                "result_summary": summarize_observations(rag_obs),
            })

        # Trace: turn end with state path and latency
        self._trace_store.append(trace_id, {
            "event": "turn_end",
            "execution_state": completed.execution_state.value,
            "state_path": [s.value for s in completed.state_history],
            "status": self._status(completed.execution_state),
            "latency_ms": (time.monotonic() - started) * 1000,
        })

        latency_ms = (time.monotonic() - started) * 1000
        for result in turn_data["results"]:
            self._stats[result.agent].record(result.success, latency_ms)

        result = CustomerTurnResult(
            trace_id=trace_id,
            response=completed.response or "抱歉，当前没有得到可用的处理结果。",
            intent=completed.dialogue_state.active_intent or "other",
            agent_type=turn_data["agent"],
            status=self._status(completed.execution_state),
            escalated=completed.execution_state == ExecutionState.FAILED,
            latency_ms=latency_ms,
            knowledge_used=any(
                logical_tool_name(observation.name) == "rag_search"
                for observation in completed.observations
            ),
            missing_slots=list(completed.dialogue_state.missing_slots),
            citations=list(turn_data["citations"]),
        )
        record_chat_turn(
            agent=result.agent_type,
            intent=result.intent,
            status=result.status,
            latency_ms=result.latency_ms,
        )
        return result

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return main-runtime Agent statistics for monitoring."""
        return {
            agent: {
                "total": stats.total,
                "success_rate": round(stats.success_rate, 3),
                "avg_ms": round(stats.avg_ms, 1),
            }
            for agent, stats in self._stats.items()
        }

    def _register_handlers(
        self,
        engine: TurnEngine,
        turn_data: Dict[str, Any],
    ) -> None:
        async def understanding(context: TurnContext) -> TurnEvent:
            if turn_data["intent_clarification"]:
                return TurnEvent(
                    type=TurnEventType.CLARIFICATION_REQUIRED,
                    response=turn_data["intent_clarification"],
                    reason="intent_clarification_required",
                )
            if context.dialogue_state.missing_slots:
                slot = context.dialogue_state.missing_slots[0]
                return TurnEvent(
                    type=TurnEventType.CLARIFICATION_REQUIRED,
                    response=self._clarification(
                        slot,
                        context.dialogue_state.active_intent,
                    ),
                    reason=f"missing_slot:{slot}",
                )
            return TurnEvent(
                type=TurnEventType.UNDERSTANDING_ACCEPTED,
                reason="understanding_complete",
            )

        async def routing(context: TurnContext) -> TurnEvent:
            if (
                context.dialogue_state.confirmation_status
                == ConfirmationStatus.REJECTED
            ):
                intent = context.dialogue_state.active_intent
                response = (
                    "已取消本次退款申请，不会执行退款操作。"
                    if intent == "refund_request"
                    else "已取消创建人工客服工单。"
                )
                return TurnEvent(
                    type=TurnEventType.ACTION_REJECTED,
                    response=response,
                    dialogue_updates={
                        "confirmation_status": ConfirmationStatus.NOT_REQUIRED,
                        "queued_goals": [],
                    },
                    reason="pending_action_rejected",
                )
            targets = self._router.route_tasks(
                context.dialogue_state,
                turn_data["intents"],
            )
            turn_data["agent"] = ",".join(targets)
            routed_event = (
                TurnEventType.ROUTED_TO_KNOWLEDGE
                if targets and all(
                    target == KNOWLEDGE_AGENT for target in targets
                )
                else TurnEventType.ROUTED_TO_ACTION
            )
            return TurnEvent(
                type=routed_event,
                dialogue_updates={
                    "last_agent": targets[-1] if targets else None,
                },
                reason=f"routed_to:{','.join(targets)}",
            )

        async def execute_agent(context: TurnContext) -> TurnEvent:
            response, results = await self._domain_runtime.execute(
                context.dialogue_state,
                turn_data["message"],
                intents=turn_data["intents"],
                context=turn_data["agent_context"],
                history=turn_data["history"],
            )
            turn_data["results"].extend(results)
            observations = [
                observation
                for result in results
                for observation in result.observations
            ]
            citations = [
                citation
                for result in results
                for citation in result.citations
            ]
            turn_data["citations"] = citations
            completed_goals = list(context.dialogue_state.completed_goals)
            for result in results:
                if not result.completed:
                    continue
                for goal in result.goals or (
                    [result.goal] if result.goal else []
                ):
                    if goal not in completed_goals:
                        completed_goals.append(goal)
            failed = next(
                (result for result in results if not result.success),
                None,
            )
            if failed is not None:
                return TurnEvent(
                    type=TurnEventType.AGENT_FAILED,
                    observations=observations,
                    response=response,
                    dialogue_updates={
                        "completed_goals": completed_goals,
                    },
                    reason=failed.error or "agent_failed",
                )
            missing = next(
                (result for result in results if result.missing_slots),
                None,
            )
            if missing is not None:
                remaining = turn_data["intents"][len(results):]
                schema = get_intent_schema(missing.goal)
                return TurnEvent(
                    type=TurnEventType.SLOTS_MISSING,
                    observations=observations,
                    response=response,
                    dialogue_updates={
                        "active_intent": missing.goal,
                        "required_slots": list(schema.required_slots),
                        "missing_slots": list(missing.missing_slots),
                        "pending_action": None,
                        "confirmation_status": ConfirmationStatus.NOT_REQUIRED,
                        "completed_goals": completed_goals,
                        "queued_goals": remaining,
                    },
                    reason="queued_goal_missing_slots",
                )
            pending_result = next(
                (result for result in results if result.pending_action is not None),
                None,
            )
            if pending_result is not None:
                remaining = turn_data["intents"][len(results):]
                schema = get_intent_schema(pending_result.goal)
                return TurnEvent(
                    type=TurnEventType.WRITE_CONFIRMATION_REQUIRED,
                    observations=observations,
                    response=response,
                    dialogue_updates={
                        "active_intent": pending_result.goal,
                        "required_slots": list(schema.required_slots),
                        "missing_slots": get_missing_slots(
                            pending_result.goal,
                            context.dialogue_state.slots,
                        ),
                        "pending_action": pending_result.pending_action,
                        "confirmation_status": ConfirmationStatus.PENDING,
                        "completed_goals": completed_goals,
                        "queued_goals": remaining,
                    },
                    reason="write_confirmation_required",
                )
            updates: Dict[str, Any] = {
                "pending_action": None,
                "confirmation_status": ConfirmationStatus.NOT_REQUIRED,
                "queued_goals": [],
                "completed_goals": completed_goals,
            }
            if self._response_polisher is not None:
                response_kind = self._response_polisher.classify(
                    results,
                    citations,
                )
                turn_data["response_kind"] = response_kind
                polish_outcome = await self._response_polisher.polish(
                    self._response_polisher.build_request(
                        response=response,
                        response_kind=response_kind,
                        observations=observations,
                        citations=citations,
                    ),
                )
                turn_data["polish_outcome"] = polish_outcome
                response = polish_outcome.response
            return TurnEvent(
                type=TurnEventType.AGENT_COMPLETED,
                observations=observations,
                response=response,
                dialogue_updates=updates,
                reason="agent_completed",
            )

        async def responding(context: TurnContext) -> TurnEvent:
            return TurnEvent(
                type=TurnEventType.RESPONSE_READY,
                response=context.response,
                reason="response_ready",
            )

        engine.register(ExecutionState.UNDERSTANDING, understanding)
        engine.register(ExecutionState.ROUTING, routing)
        engine.register(ExecutionState.RETRIEVING, execute_agent)
        engine.register(ExecutionState.ACTING, execute_agent)
        engine.register(ExecutionState.RESPONDING, responding)

    @staticmethod
    def _entry_state(
        engine: TurnEngine,
        dialogue_state,
    ) -> ExecutionState:
        """Map persisted business state to an executable state for this turn."""
        resumed = engine.resume_state(dialogue_state)
        if resumed == ExecutionState.CLARIFYING:
            return ExecutionState.UNDERSTANDING
        if resumed == ExecutionState.AWAITING_CONFIRMATION:
            return ExecutionState.ACTING
        if resumed == ExecutionState.RESPONDING:
            return ExecutionState.ROUTING
        return resumed

    @staticmethod
    def _clarification(slot: str, intent: Optional[str] = None) -> str:
        if intent == "logistics_query":
            return "请提供订单号或物流单号。"
        prompts = {
            "order_id": "请提供需要处理的订单号。",
            "tracking_no": "请提供物流单号。",
        }
        return prompts.get(slot, f"请补充 {slot}。")

    @classmethod
    def _intent_clarification(cls, state, understanding) -> Optional[str]:
        """Return a clarification prompt when an intent is unsafe to accept."""
        if (
            understanding.user_act in {UserAct.CONFIRM, UserAct.REJECT}
            and state.confirmation_status == ConfirmationStatus.PENDING
        ):
            return None
        if (
            state.pending_action is not None
            and understanding.primary_intent
            not in {state.active_intent, "other"}
            and understanding.user_act != UserAct.SWITCH
        ):
            return (
                f"当前 {state.active_intent} 操作正在等待确认。"
                f"是否取消并切换到 {understanding.primary_intent}？"
            )
        action_intent = understanding.primary_intent in cls._ACTION_INTENTS
        threshold_name = (
            "NLU_ACTION_MIN_CONFIDENCE"
            if action_intent
            else "NLU_MIN_CONFIDENCE"
        )
        default = "0.85" if action_intent else "0.65"
        threshold = float(os.getenv(threshold_name, default))
        if understanding.confidence >= threshold:
            return None
        return "我还不能确定你的需求，请说明要查询订单、物流，还是办理售后。"

    @staticmethod
    def _status(state: ExecutionState) -> str:
        if state in {
            ExecutionState.CLARIFYING,
            ExecutionState.AWAITING_CONFIRMATION,
        }:
            return "awaiting_user"
        if state == ExecutionState.FAILED:
            return "failed"
        return "completed"
