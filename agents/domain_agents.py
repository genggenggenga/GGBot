"""Deterministic domain agents built on the shared ToolRegistry."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from core.agent_models import (
    ConfirmationStatus,
    DialogueState,
    Observation,
    PendingAction,
)
from core.tool_registry import ToolRegistry


KNOWLEDGE_AGENT = "knowledge"
ORDER_AGENT = "order"
LOGISTICS_AGENT = "logistics"
AFTER_SALES_AGENT = "after_sales"


@dataclass
class AgentResult:
    agent: str
    success: bool
    response: str
    observations: List[Observation] = field(default_factory=list)
    pending_action: Optional[PendingAction] = None
    completed: bool = True
    citations: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


class Router:
    """Route from explicit dialogue state without another LLM call."""

    _INTENT_ROUTES = {
        "order_query": ORDER_AGENT,
        "logistics_query": LOGISTICS_AGENT,
        "refund_request": AFTER_SALES_AGENT,
        "return_request": AFTER_SALES_AGENT,
        "cancel_order": AFTER_SALES_AGENT,
        "complaint": AFTER_SALES_AGENT,
        "escalation": AFTER_SALES_AGENT,
        "refund_policy": KNOWLEDGE_AGENT,
        "query": KNOWLEDGE_AGENT,
        "technical": KNOWLEDGE_AGENT,
        "account": KNOWLEDGE_AGENT,
        "greeting": KNOWLEDGE_AGENT,
        "feedback": KNOWLEDGE_AGENT,
        "other": KNOWLEDGE_AGENT,
    }

    def route(self, state: DialogueState) -> str:
        intent = state.active_intent or "other"
        if intent == "billing":
            return (
                AFTER_SALES_AGENT
                if state.pending_action is not None
                else KNOWLEDGE_AGENT
            )
        return self._INTENT_ROUTES.get(intent, KNOWLEDGE_AGENT)

    def route_tasks(
        self,
        state: DialogueState,
        intents: Optional[Sequence[str]] = None,
    ) -> List[str]:
        targets = [
            self._INTENT_ROUTES.get(intent, self.route(state))
            for intent in (intents or [state.active_intent or "other"])
        ]
        return list(dict.fromkeys(targets))


class ResponseComposer:
    """Compose sequential sub-task results without exposing hidden reasoning."""

    @staticmethod
    def compose(results: Sequence[AgentResult]) -> str:
        responses = [result.response.strip() for result in results if result.response.strip()]
        if responses:
            return "\n\n".join(responses)
        return "抱歉，当前没有得到可用的处理结果。"


class ServiceAgent:
    """Shared bounded execution for order, logistics and after-sales agents."""

    name = "service"
    system_prompt = ""
    allowed_tools: tuple[str, ...] = ()
    completion_condition = ""

    def __init__(self, registry: ToolRegistry, max_steps: int = 4) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self._registry = registry
        self._max_steps = max_steps
        self._registry.set_agent_whitelist(self.name, set(self.allowed_tools))

    async def execute(
        self,
        state: DialogueState,
        message: str = "",
        context: str = "",
    ) -> AgentResult:
        observations: List[Observation] = []
        for _ in range(self._max_steps):
            action = self.next_action(state, message, observations)
            if action is None:
                return self.finish(state, observations)
            tool_name, params, action_id = action
            result = await self._registry.call(
                self.name,
                tool_name,
                params,
                context={"prompt_context": context} if context else None,
                action_id=action_id,
            )
            observation = result.to_observation()
            observations.append(observation)
            if not result.success:
                return AgentResult(
                    agent=self.name,
                    success=False,
                    response=self.failure_response(tool_name, result.error),
                    observations=observations,
                    completed=False,
                    error=result.error,
                )

        if self.next_action(state, message, observations) is not None:
            return AgentResult(
                agent=self.name,
                success=False,
                response="处理步骤超过限制，建议转人工继续处理。",
                observations=observations,
                completed=False,
                error="max_steps_exceeded",
            )
        return self.finish(state, observations)

    def next_action(
        self,
        state: DialogueState,
        message: str,
        observations: Sequence[Observation],
    ) -> Optional[tuple[str, Dict[str, Any], Optional[str]]]:
        raise NotImplementedError

    def finish(
        self,
        state: DialogueState,
        observations: List[Observation],
    ) -> AgentResult:
        raise NotImplementedError

    def failure_response(self, tool_name: str, error: Optional[str]) -> str:
        del error
        return f"{tool_name} 调用失败，暂时无法完成当前请求。"


class OrderAgent(ServiceAgent):
    name = ORDER_AGENT
    system_prompt = "只根据订单和支付工具返回的事实回答，不猜测订单状态。"
    allowed_tools = ("query_order", "query_payment")
    completion_condition = "order_fact_returned"

    def next_action(self, state, message, observations):
        del message
        if observations:
            return None
        order_id = state.slots.get("order_id")
        return "query_order", {"order_id": order_id}, None

    def finish(self, state, observations):
        del state
        order = observations[-1].data
        return AgentResult(
            agent=self.name,
            success=True,
            response=f"订单 {order.get('order_id')} 当前状态：{order.get('status')}。",
            observations=observations,
        )


class LogisticsAgent(ServiceAgent):
    name = LOGISTICS_AGENT
    system_prompt = "结合订单和物流轨迹解释配送状态，不编造预计时间。"
    allowed_tools = ("query_order", "track_package", "rag_search")
    completion_condition = "logistics_fact_returned"

    def next_action(self, state, message, observations):
        del message
        order_id = state.slots.get("order_id")
        if not observations:
            return "query_order", {"order_id": order_id}, None
        if len(observations) == 1 and observations[0].data.get("found", True):
            return "track_package", {"order_id": order_id}, None
        return None

    def finish(self, state, observations):
        del state
        if observations[-1].name == "query_order":
            return AgentResult(
                agent=self.name,
                success=True,
                response="没有找到对应订单，暂时无法查询物流。",
                observations=observations,
            )
        logistics = observations[-1].data
        status = logistics.get("status") or logistics.get("current_status")
        return AgentResult(
            agent=self.name,
            success=True,
            response=f"当前物流状态：{status}。",
            observations=observations,
        )


class AfterSalesAgent(ServiceAgent):
    name = AFTER_SALES_AGENT
    system_prompt = "先核验订单和售后资格；创建退款前必须取得用户确认。"
    allowed_tools = (
        "query_order",
        "check_refund_eligibility",
        "create_refund",
        "create_ticket",
        "rag_search",
    )
    completion_condition = "refund_created_or_handoff_created"

    async def execute(
        self,
        state: DialogueState,
        message: str = "",
        context: str = "",
    ) -> AgentResult:
        if (
            state.pending_action is not None
            and state.confirmation_status == ConfirmationStatus.PENDING
        ):
            return AgentResult(
                agent=self.name,
                success=True,
                response="请确认是否提交退款申请。",
                pending_action=state.pending_action,
                completed=False,
            )
        return await super().execute(state, message, context)

    def next_action(self, state, message, observations):
        del message
        order_id = state.slots.get("order_id")
        pending = state.pending_action
        if (
            pending is not None
            and state.confirmation_status == ConfirmationStatus.CONFIRMED
        ):
            self._registry.confirm_action(pending.action_id)
            arguments = dict(pending.arguments)
            arguments.setdefault("action_id", pending.action_id)
            return (
                None
                if observations
                else (pending.tool_name, arguments, pending.action_id)
            )
        if not observations:
            return "query_order", {"order_id": order_id}, None
        if len(observations) == 1 and observations[0].data.get("found", True):
            return (
                "check_refund_eligibility",
                {"order_id": order_id},
                None,
            )
        return None

    def finish(self, state, observations):
        pending = state.pending_action
        if pending is not None and observations[-1].name == "create_refund":
            refund = observations[-1].data
            return AgentResult(
                agent=self.name,
                success=True,
                response=f"退款申请已提交，申请编号：{refund.get('refund_id')}。",
                observations=observations,
            )

        if observations[-1].name == "query_order":
            return AgentResult(
                agent=self.name,
                success=True,
                response="没有找到对应订单，暂时无法判断退款资格。",
                observations=observations,
            )
        eligibility = observations[-1].data
        if not eligibility.get("eligible", False):
            reason = eligibility.get("reason", "当前订单不满足退款条件")
            return AgentResult(
                agent=self.name,
                success=True,
                response=f"暂时无法发起退款：{reason}。",
                observations=observations,
            )

        action = PendingAction(
            tool_name="create_refund",
            arguments={"order_id": state.slots.get("order_id")},
        )
        self._registry.mark_pending_action(action.action_id)
        return AgentResult(
            agent=self.name,
            success=True,
            response="订单符合退款条件。请确认是否提交退款申请。",
            observations=observations,
            pending_action=action,
            completed=False,
        )


class KnowledgeAgent:
    """Fixed one-shot RAG agent; it never enters the ReAct loop."""

    name = KNOWLEDGE_AGENT
    allowed_tools = ("rag_search",)

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._registry.set_agent_whitelist(self.name, set(self.allowed_tools))

    async def execute(
        self,
        state: DialogueState,
        message: str,
        context: str = "",
    ) -> AgentResult:
        del state
        result = await self._registry.call(
            self.name,
            "rag_search",
            {"query": message, "mode": "rerank"},
            context={"prompt_context": context} if context else None,
        )
        observation = result.to_observation()
        if not result.success:
            return AgentResult(
                agent=self.name,
                success=False,
                response="现有资料暂时无法回答该问题。",
                observations=[observation],
                error=result.error,
            )

        payload = result.data or {}
        items = payload.get("hits") or payload.get("items") or payload.get("results") or []
        citations = payload.get("citations") or []
        if payload.get("answered") is False or not items:
            return AgentResult(
                agent=self.name,
                success=True,
                response="根据现有资料无法回答该问题。",
                observations=[observation],
                citations=citations,
            )

        first = items[0]
        chunk = first.get("chunk") if isinstance(first, dict) else None
        content = (
            (chunk or {}).get("content")
            or first.get("content")
            or first.get("text")
            or str(first)
        )
        return AgentResult(
            agent=self.name,
            success=True,
            response=f"{content} [1]",
            observations=[observation],
            citations=citations,
        )


class DomainAgentRuntime:
    """Execute one or more routed tasks sequentially."""

    def __init__(
        self,
        router: Router,
        agents: Dict[str, Any],
        composer: Optional[ResponseComposer] = None,
    ) -> None:
        self._router = router
        self._agents = agents
        self._composer = composer or ResponseComposer()

    async def execute(
        self,
        state: DialogueState,
        message: str,
        intents: Optional[Sequence[str]] = None,
        context: str = "",
    ) -> tuple[str, List[AgentResult]]:
        results = []
        for target in self._router.route_tasks(state, intents):
            agent = self._agents[target]
            if context:
                results.append(await agent.execute(state, message, context))
            else:
                results.append(await agent.execute(state, message))
        return self._composer.compose(results), results
