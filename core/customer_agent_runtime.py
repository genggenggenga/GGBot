"""State-driven customer-service runtime used by the /chat endpoint."""
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.domain_agents import (
    KNOWLEDGE_AGENT,
    AgentResult,
    DomainAgentRuntime,
    Router,
)
from core.agent_models import (
    ConfirmationStatus,
    ExecutionState,
    Transition,
    TurnContext,
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
        execution = TurnContext(
            user_id=user_id,
            conv_id=conv_id,
            dialogue_state=dialogue_state,
            execution_state=ExecutionState.UNDERSTANDING,
        )
        turn_data: Dict[str, Any] = {
            "message": message,
            "agent_context": agent_context,
            "agent": self._router.route(dialogue_state),
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

        return CustomerTurnResult(
            trace_id=trace_id,
            response=completed.response or "抱歉，当前没有得到可用的处理结果。",
            intent=completed.dialogue_state.active_intent or "other",
            agent_type=turn_data["agent"],
            status=self._status(completed.execution_state),
            escalated=completed.execution_state == ExecutionState.FAILED,
            latency_ms=(time.monotonic() - started) * 1000,
            knowledge_used=any(
                observation.name == "rag_search"
                for observation in completed.observations
            ),
            missing_slots=list(completed.dialogue_state.missing_slots),
            citations=list(turn_data["citations"]),
        )

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
                return Transition(
                    next_state=ExecutionState.RESPONDING,
                    response="已取消本次退款申请，不会执行退款操作。",
                    dialogue_updates={
                        "confirmation_status": ConfirmationStatus.NOT_REQUIRED,
                    },
                    reason="pending_action_rejected",
                )
            target = self._router.route(context.dialogue_state)
            turn_data["agent"] = target
            return Transition(
                next_state=(
                    ExecutionState.RETRIEVING
                    if target == KNOWLEDGE_AGENT
                    else ExecutionState.ACTING
                ),
                dialogue_updates={"last_agent": target},
                reason=f"routed_to:{target}",
            )

        async def execute_agent(context: TurnContext) -> Transition:
            _, results = await self._domain_runtime.execute(
                context.dialogue_state,
                turn_data["message"],
                context=turn_data["agent_context"],
            )
            turn_data["results"].extend(results)
            result: AgentResult = results[-1]
            turn_data["citations"] = result.citations
            if not result.success:
                return Transition(
                    next_state=ExecutionState.FAILED,
                    observations=result.observations,
                    response=result.response,
                    reason=result.error or "agent_failed",
                )
            if result.pending_action is not None:
                return Transition(
                    next_state=ExecutionState.AWAITING_CONFIRMATION,
                    observations=result.observations,
                    response=result.response,
                    dialogue_updates={
                        "pending_action": result.pending_action,
                        "confirmation_status": ConfirmationStatus.PENDING,
                    },
                    reason="write_confirmation_required",
                )
            updates: Dict[str, Any] = {
                "pending_action": None,
                "confirmation_status": ConfirmationStatus.NOT_REQUIRED,
            }
            if result.completed:
                goals = list(context.dialogue_state.completed_goals)
                goal = context.dialogue_state.active_intent
                if goal and goal not in goals:
                    goals.append(goal)
                updates["completed_goals"] = goals
            return Transition(
                next_state=ExecutionState.RESPONDING,
                observations=result.observations,
                response=result.response,
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
