"""State-driven customer-service runtime used by the /chat endpoint."""
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
    Transition,
    TurnContext,
    UserAct,
)
from core.dialogue_state_tracker import DialogueStateTracker
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

    def __init__(
        self,
        recognizer: Any,
        tracker: DialogueStateTracker,
        turn_engine: TurnEngine,
        domain_runtime: DomainAgentRuntime,
        router: Router,
        trace_store: TraceStore,
    ) -> None:
        self._recognizer = recognizer
        self._tracker = tracker
        self._turn_engine = turn_engine
        self._domain_runtime = domain_runtime
        self._router = router
        self._trace_store = trace_store
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
        turn_data: Dict[str, Any] = {
            "message": message,
            "agent_context": agent_context,
            "agent": self._router.route(dialogue_state),
            "intents": execution_intents,
            "results": [],
            "citations": [],
        }
        self._register_handlers(engine, turn_data)

        # Trace: understanding phase - only intent/slot summaries, no user text
        self._trace_store.append(trace_id, {
            "event": "understanding",
            "intent": understanding.primary_intent,
            "slot_keys": list(understanding.extracted_slots),
            "user_act": understanding.user_act.value,
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

        # Trace: RAG-specific summary if rag_search was used
        rag_obs = [
            obs for obs in completed.observations
            if obs.name == "rag_search" and obs.success
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

        return CustomerTurnResult(
            trace_id=trace_id,
            response=completed.response or "抱歉，当前没有得到可用的处理结果。",
            intent=completed.dialogue_state.active_intent or "other",
            agent_type=turn_data["agent"],
            status=self._status(completed.execution_state),
            escalated=completed.execution_state == ExecutionState.FAILED,
            latency_ms=latency_ms,
            knowledge_used=any(
                observation.name == "rag_search"
                for observation in completed.observations
            ),
            missing_slots=list(completed.dialogue_state.missing_slots),
            citations=list(turn_data["citations"]),
        )

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return main-runtime Agent statistics for monitoring."""
        return {
            agent: {
                "total": stats.total,
                "success_rate": round(stats.success_rate, 3),
                "avg_ms": round(stats.avg_ms, 1),
                "routing_score": round(stats.success_rate, 3),
            }
            for agent, stats in self._stats.items()
        }

    def _register_handlers(
        self,
        engine: TurnEngine,
        turn_data: Dict[str, Any],
    ) -> None:
        async def understanding(context: TurnContext) -> Transition:
            if context.dialogue_state.missing_slots:
                slot = context.dialogue_state.missing_slots[0]
                return Transition(
                    next_state=ExecutionState.CLARIFYING,
                    response=self._clarification(slot),
                    reason=f"missing_slot:{slot}",
                )
            return Transition(
                next_state=ExecutionState.ROUTING,
                reason="understanding_complete",
            )

        async def routing(context: TurnContext) -> Transition:
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
                return Transition(
                    next_state=ExecutionState.RESPONDING,
                    response=response,
                    dialogue_updates={
                        "confirmation_status": ConfirmationStatus.NOT_REQUIRED,
                    },
                    reason="pending_action_rejected",
                )
            targets = self._router.route_tasks(
                context.dialogue_state,
                turn_data["intents"],
            )
            turn_data["agent"] = ",".join(targets)
            return Transition(
                next_state=(
                    ExecutionState.RETRIEVING
                    if targets and all(
                        target == KNOWLEDGE_AGENT for target in targets
                    )
                    else ExecutionState.ACTING
                ),
                dialogue_updates={
                    "last_agent": targets[-1] if targets else None,
                },
                reason=f"routed_to:{','.join(targets)}",
            )

        async def execute_agent(context: TurnContext) -> Transition:
            response, results = await self._domain_runtime.execute(
                context.dialogue_state,
                turn_data["message"],
                intents=turn_data["intents"],
                context=turn_data["agent_context"],
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
            failed = next(
                (result for result in results if not result.success),
                None,
            )
            if failed is not None:
                return Transition(
                    next_state=ExecutionState.FAILED,
                    observations=observations,
                    response=response,
                    reason=failed.error or "agent_failed",
                )
            pending = next(
                (
                    result.pending_action
                    for result in results
                    if result.pending_action is not None
                ),
                None,
            )
            if pending is not None:
                return Transition(
                    next_state=ExecutionState.AWAITING_CONFIRMATION,
                    observations=observations,
                    response=response,
                    dialogue_updates={
                        "pending_action": pending,
                        "confirmation_status": ConfirmationStatus.PENDING,
                    },
                    reason="write_confirmation_required",
                )
            updates: Dict[str, Any] = {
                "pending_action": None,
                "confirmation_status": ConfirmationStatus.NOT_REQUIRED,
            }
            goals = list(context.dialogue_state.completed_goals)
            for result in results:
                if not result.completed:
                    continue
                result_goals = result.goals or (
                    [result.goal] if result.goal else []
                )
                for goal in result_goals:
                    if goal not in goals:
                        goals.append(goal)
            updates["completed_goals"] = goals
            return Transition(
                next_state=ExecutionState.RESPONDING,
                observations=observations,
                response=response,
                dialogue_updates=updates,
                reason="agent_completed",
            )

        async def responding(context: TurnContext) -> Transition:
            return Transition(
                next_state=ExecutionState.COMPLETED,
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
    def _clarification(slot: str) -> str:
        prompts = {
            "order_id": "请提供需要处理的订单号。",
            "tracking_no": "请提供物流单号。",
        }
        return prompts.get(slot, f"请补充 {slot}。")

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
