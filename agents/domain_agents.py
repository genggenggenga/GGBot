"""Deterministic domain agents built on the shared ToolRegistry."""
from dataclasses import dataclass, field
import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict

from core.agent_models import (
    ConfirmationStatus,
    DecisionType,
    DialogueState,
    Observation,
    PendingAction,
    get_intent_schema,
    get_missing_slots,
)
from core.prompts.react import (
    AFTER_SALES_DOMAIN_POLICY,
    LOGISTICS_DOMAIN_POLICY,
    ORDER_DOMAIN_POLICY,
)
from core.react_planner import ReActPlanner
from core.tool_names import canonical_tool_name, logical_tool_name
from core.tool_registry import ToolRegistry, ToolType
from rag.answer_generator import RAGAnswerGenerator
from rag.query_planner import QueryPlanner


logger = logging.getLogger(__name__)

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
    goal: Optional[str] = None
    goals: List[str] = field(default_factory=list)
    missing_slots: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentTask:
    goal: str
    target_agent: str


class OrderQueryOutput(BaseModel):
    """Business result returned by query_order."""

    model_config = ConfigDict(extra="allow")

    found: bool = True
    order_id: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None


class LogisticsQueryOutput(BaseModel):
    """Business result returned by track_package."""

    model_config = ConfigDict(extra="allow")

    found: bool = True
    status: Optional[str] = None
    current_status: Optional[str] = None
    error: Optional[str] = None


class RefundCreateOutput(BaseModel):
    """Business result returned by create_refund."""

    model_config = ConfigDict(extra="allow")

    created: bool
    refund_id: Optional[str] = None
    error: Optional[str] = None


class TicketCreateOutput(BaseModel):
    """Business result returned by create_ticket."""

    model_config = ConfigDict(extra="allow")

    created: bool
    ticket_id: Optional[str] = None
    error: Optional[str] = None


class Router:
    """Route from explicit dialogue state without another LLM call."""

    _INTENT_ROUTES = {
        "order_query": ORDER_AGENT,
        "logistics_query": LOGISTICS_AGENT,
        "refund_request": AFTER_SALES_AGENT,
        "return_request": AFTER_SALES_AGENT,
        "cancel_order": AFTER_SALES_AGENT,
        "request": AFTER_SALES_AGENT,
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
        return self.route_intent(state.active_intent or "other", state)

    def route_intent(self, intent: str, state: DialogueState) -> str:
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
            self.route_intent(intent, state)
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


class GoalCompletionEvaluator:
    """Evaluate executable completion conditions declared by IntentSchema."""

    @staticmethod
    def is_complete(intent: str, result: AgentResult) -> bool:
        if not result.success or result.pending_action is not None:
            return False
        condition = get_intent_schema(intent).completion_condition
        observations = result.observations
        names = {
            logical_tool_name(observation.name)
            for observation in observations
        }

        if condition in {"response_generated", "response_or_clarification_generated"}:
            return True
        if condition == "answer_grounded_in_knowledge":
            return bool(result.citations)
        if condition == "order_fact_returned":
            return bool(names & {
                "query_order",
                "query_order_items",
                "query_payment_detail",
                "query_invoice",
            })
        if condition == "logistics_fact_returned":
            return bool(names & {
                "query_order",
                "track_package",
                "estimate_delivery",
                "diagnose_delivery_exception",
                "rag_search",
            })
        if condition == "refund_created_or_handoff_created":
            return bool(names & {
                "create_refund",
                "check_refund_eligibility",
                "evaluate_after_sales_options",
                "calculate_refund_quote",
                "query_order",
            })
        if condition == "return_created_or_handoff_created":
            return bool(names & {
                "create_return",
                "evaluate_after_sales_options",
                "create_ticket",
            })
        if condition == "order_cancelled_or_handoff_created":
            return bool(names & {
                "cancel_order",
                "evaluate_after_sales_options",
                "create_ticket",
            })
        if condition == "handoff_created":
            return any(
                logical_tool_name(observation.name) == "create_ticket"
                and isinstance(observation.data, dict)
                and observation.data.get("created") is True
                for observation in observations
            )
        return result.completed


class ServiceAgent:
    """Shared deterministic fallback and bounded ReAct execution."""

    name = "service"
    system_prompt = ""
    allowed_tools: tuple[str, ...] = ()
    react_allowed_tools: tuple[str, ...] = ()
    deterministic_fallback_intents: Optional[tuple[str, ...]] = None
    completion_condition = ""

    def __init__(
        self,
        registry: ToolRegistry,
        max_steps: int = 4,
        planner: Optional[ReActPlanner] = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self._registry = registry
        self._max_steps = max_steps
        self._planner = planner
        self._tool_names = {
            name: (
                canonical_tool_name(name)
                if registry.get_spec(canonical_tool_name(name)) is not None
                else name
            )
            for name in self.allowed_tools
        }
        self._resolved_allowed_tools = set(self._tool_names.values())
        self._registry.set_agent_whitelist(
            self.name,
            self._resolved_allowed_tools,
        )

    def tool_name(self, logical_name: str) -> str:
        """Resolve a logical tool to its registered domain-qualified name."""
        return self._tool_names.get(logical_name, logical_name)

    async def execute(
        self,
        state: DialogueState,
        message: str = "",
        context: str = "",
        goal: Optional[str] = None,
        seed_observations: Optional[Sequence[Observation]] = None,
        skill_context: str = "",
    ) -> AgentResult:
        task_state = (
            state.model_copy(update={"active_intent": goal}, deep=True)
            if goal
            else state
        )
        observations = list(seed_observations or [])
        if self._planner is not None:
            return await self._execute_react(
                task_state,
                message,
                context,
                observations,
                skill_context,
            )
        return await self._execute_deterministic(
            task_state,
            message,
            context,
            observations,
        )

    async def _execute_deterministic(
        self,
        task_state: DialogueState,
        message: str,
        context: str,
        observations: List[Observation],
    ) -> AgentResult:
        for _ in range(self._max_steps):
            action = self.next_action(task_state, message, observations)
            if action is None:
                return self.finish(task_state, observations)
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

        if self.next_action(task_state, message, observations) is not None:
            return AgentResult(
                agent=self.name,
                success=False,
                response="处理步骤超过限制，建议转人工继续处理。",
                observations=observations,
                completed=False,
                error="max_steps_exceeded",
            )
        return self.finish(task_state, observations)

    async def _execute_react(
        self,
        task_state: DialogueState,
        message: str,
        context: str,
        observations: List[Observation],
        skill_context: str = "",
    ) -> AgentResult:
        called_actions = set()
        pending = task_state.pending_action
        if pending is not None:
            if task_state.confirmation_status == ConfirmationStatus.PENDING:
                return AgentResult(
                    agent=self.name,
                    success=True,
                    response=f"请确认是否执行 {pending.tool_name}。",
                    observations=observations,
                    pending_action=pending,
                    completed=False,
                )
            if task_state.confirmation_status == ConfirmationStatus.CONFIRMED:
                if pending.tool_name not in self._resolved_allowed_tools:
                    return AgentResult(
                        agent=self.name,
                        success=False,
                        response="待确认操作不属于当前 Agent。",
                        observations=observations,
                        completed=False,
                        error="pending_tool_not_allowed",
                    )
                self._registry.confirm_action(pending.action_id)
                arguments = dict(pending.arguments)
                arguments["action_id"] = pending.action_id
                result = await self._registry.call(
                    self.name,
                    pending.tool_name,
                    arguments,
                    context={"prompt_context": context} if context else None,
                    action_id=pending.action_id,
                )
                observation = result.to_observation()
                observations.append(observation)
                called_actions.add(self._action_key(
                    pending.tool_name,
                    pending.arguments,
                ))
                if not result.success:
                    return AgentResult(
                        agent=self.name,
                        success=False,
                        response=self.failure_response(
                            pending.tool_name,
                            result.error,
                        ),
                        observations=observations,
                        completed=False,
                        error=result.error,
                    )

        react_tools = {
            self.tool_name(name)
            for name in (self.react_allowed_tools or self.allowed_tools)
        }
        tool_specs = [
            spec
            for spec in self._registry.list_tools(self.name)
            if spec.name in react_tools
        ]
        for _ in range(self._max_steps):
            try:
                decision = await self._planner.decide(
                    agent_name=self.name,
                    goal=task_state.active_intent or "other",
                    message=message,
                    state=task_state,
                    observations=observations,
                    tools=tool_specs,
                    system_prompt=self.system_prompt,
                    skill_context=skill_context,
                )
            except Exception as ex:
                logger.warning(
                    "ReAct planning failed for %s, using deterministic fallback: %s",
                    self.name,
                    ex,
                )
                if (
                    self.deterministic_fallback_intents is not None
                    and task_state.active_intent
                    not in self.deterministic_fallback_intents
                ):
                    return AgentResult(
                        agent=self.name,
                        success=True,
                        response=(
                            "当前售后自动处理暂不可用，"
                            "建议转人工客服继续处理。"
                        ),
                        observations=observations,
                        completed=False,
                    )
                return await self._execute_deterministic(
                    task_state,
                    message,
                    context,
                    [],
                )

            if decision.type == DecisionType.FINISH:
                return AgentResult(
                    agent=self.name,
                    success=True,
                    response=decision.response or "",
                    observations=observations,
                )
            if decision.type == DecisionType.CLARIFY:
                return AgentResult(
                    agent=self.name,
                    success=True,
                    response=decision.response or "请补充必要信息。",
                    observations=observations,
                    completed=False,
                    missing_slots=list(task_state.missing_slots),
                )
            if decision.type == DecisionType.HANDOFF:
                return AgentResult(
                    agent=self.name,
                    success=True,
                    response=decision.response or "建议转人工继续处理。",
                    observations=observations,
                    completed=False,
                )

            tool_name = decision.tool_name or ""
            spec = self._registry.get_spec(tool_name)
            if spec is None or tool_name not in react_tools:
                return AgentResult(
                    agent=self.name,
                    success=False,
                    response="Agent 选择了未授权工具，已停止执行。",
                    observations=observations,
                    completed=False,
                    error=f"react_tool_not_allowed:{tool_name}",
                )
            arguments = dict(decision.arguments)
            arguments.pop("action_id", None)
            action_key = self._action_key(tool_name, arguments)
            if action_key in called_actions:
                return AgentResult(
                    agent=self.name,
                    success=False,
                    response="检测到重复工具调用，已停止执行。",
                    observations=observations,
                    completed=False,
                    error="repeated_tool_call",
                )
            called_actions.add(action_key)

            if spec.tool_type == ToolType.WRITE:
                action = PendingAction(
                    tool_name=tool_name,
                    arguments=arguments,
                )
                self._registry.mark_pending_action(action.action_id)
                return AgentResult(
                    agent=self.name,
                    success=True,
                    response=(
                        decision.response
                        or f"即将执行 {tool_name}，请确认是否继续。"
                    ),
                    observations=observations,
                    pending_action=action,
                    completed=False,
                )

            result = await self._registry.call(
                self.name,
                tool_name,
                arguments,
                context={"prompt_context": context} if context else None,
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

        return AgentResult(
            agent=self.name,
            success=False,
            response="处理步骤超过限制，建议转人工继续处理。",
            observations=observations,
            completed=False,
            error="max_steps_exceeded",
        )

    @staticmethod
    def _action_key(tool_name: str, arguments: Any) -> str:
        return f"{tool_name}:{json.dumps(arguments, sort_keys=True, default=str)}"

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
    system_prompt = ORDER_DOMAIN_POLICY
    allowed_tools = (
        "query_order",
        "query_order_items",
        "query_payment_detail",
        "query_invoice",
    )
    completion_condition = "order_fact_returned"

    def next_action(self, state, message, observations):
        del message
        if observations:
            return None
        order_id = state.slots.get("order_id")
        return self.tool_name("query_order"), {"order_id": order_id}, None

    def finish(self, state, observations):
        del state
        order = OrderQueryOutput.model_validate(observations[-1].data or {})
        if not order.found:
            return AgentResult(
                agent=self.name,
                success=True,
                response=f"没有找到订单 {order.order_id or ''}，请核对订单号。",
                observations=observations,
            )
        if not order.status:
            return AgentResult(
                agent=self.name,
                success=False,
                response="订单查询结果缺少状态，暂时无法回答。",
                observations=observations,
                completed=False,
                error="invalid_order_result:missing_status",
            )
        return AgentResult(
            agent=self.name,
            success=True,
            response=f"订单 {order.order_id} 当前状态：{order.status}。",
            observations=observations,
        )


class LogisticsAgent(ServiceAgent):
    name = LOGISTICS_AGENT
    system_prompt = LOGISTICS_DOMAIN_POLICY
    allowed_tools = (
        "query_order",
        "track_package",
        "estimate_delivery",
        "diagnose_delivery_exception",
        "rag_search",
    )
    completion_condition = "logistics_fact_returned"

    def next_action(self, state, message, observations):
        del message
        order_id = state.slots.get("order_id")
        tracking_no = state.slots.get("tracking_no")
        if not observations:
            if tracking_no and not order_id:
                return (
                    self.tool_name("track_package"),
                    {"tracking_no": tracking_no},
                    None,
                )
            return self.tool_name("query_order"), {"order_id": order_id}, None
        if (
            len(observations) == 1
            and logical_tool_name(observations[0].name) == "query_order"
            and observations[0].data.get("found", True)
        ):
            return (
                self.tool_name("track_package"),
                {"order_id": order_id},
                None,
            )
        return None

    def finish(self, state, observations):
        del state
        if logical_tool_name(observations[-1].name) == "query_order":
            return AgentResult(
                agent=self.name,
                success=True,
                response="没有找到对应订单，暂时无法查询物流。",
                observations=observations,
            )
        logistics = LogisticsQueryOutput.model_validate(
            observations[-1].data or {},
        )
        if not logistics.found:
            return AgentResult(
                agent=self.name,
                success=True,
                response="该订单暂无物流信息。",
                observations=observations,
            )
        status = logistics.status or logistics.current_status
        if not status:
            return AgentResult(
                agent=self.name,
                success=False,
                response="物流查询结果缺少状态，暂时无法回答。",
                observations=observations,
                completed=False,
                error="invalid_logistics_result:missing_status",
            )
        return AgentResult(
            agent=self.name,
            success=True,
            response=f"当前物流状态：{status}。",
            observations=observations,
        )


class AfterSalesAgent(ServiceAgent):
    name = AFTER_SALES_AGENT
    system_prompt = AFTER_SALES_DOMAIN_POLICY
    deterministic_fallback_intents = ("refund_request",)
    allowed_tools = (
        "query_order",
        "check_refund_eligibility",
        "evaluate_after_sales_options",
        "calculate_refund_quote",
        "create_refund",
        "create_return",
        "cancel_order",
        "create_ticket",
        "rag_search",
    )
    react_allowed_tools = (
        "query_order",
        "evaluate_after_sales_options",
        "calculate_refund_quote",
        "create_refund",
        "create_return",
        "cancel_order",
        "create_ticket",
    )
    completion_condition = "refund_created_or_handoff_created"
    _UNIMPLEMENTED_RESPONSES = {
        "return_request": "当前版本尚未实现退货操作，请转人工客服继续处理。",
        "cancel_order": "当前版本尚未实现取消订单操作，请转人工客服继续处理。",
        "request": "当前版本尚未实现该售后操作，请转人工客服继续处理。",
    }

    def __init__(
        self,
        registry: ToolRegistry,
        max_steps: int = 4,
        planner: Optional[ReActPlanner] = None,
        skill_manager: Optional[Any] = None,
    ) -> None:
        super().__init__(registry, max_steps=max_steps, planner=planner)
        self._skill_manager = skill_manager

    async def execute(
        self,
        state: DialogueState,
        message: str = "",
        context: str = "",
        goal: Optional[str] = None,
        seed_observations: Optional[Sequence[Observation]] = None,
    ) -> AgentResult:
        task_state = (
            state.model_copy(update={"active_intent": goal}, deep=True)
            if goal
            else state
        )
        skill_context = (
            self._skill_manager.prompt_for(
                message,
                agent_type=self.name,
                intent=task_state.active_intent,
            )
            if self._skill_manager is not None
            else ""
        )
        if task_state.active_intent in {"complaint", "escalation"}:
            return await self._execute_handoff(task_state, message, context)
        if (
            task_state.active_intent != "refund_request"
            and self._planner is None
        ):
            return AgentResult(
                agent=self.name,
                success=True,
                response=self._UNIMPLEMENTED_RESPONSES.get(
                    task_state.active_intent or "",
                    "当前版本尚未实现该售后操作，请转人工客服继续处理。",
                ),
                completed=False,
            )
        if (
            task_state.pending_action is not None
            and task_state.confirmation_status == ConfirmationStatus.PENDING
        ):
            return AgentResult(
                agent=self.name,
                success=True,
                response="请确认是否提交退款申请。",
                pending_action=task_state.pending_action,
                completed=False,
            )
        return await super().execute(
            task_state,
            message,
            context,
            seed_observations=seed_observations,
            skill_context=skill_context,
        )

    async def _execute_handoff(
        self,
        state: DialogueState,
        message: str,
        context: str,
    ) -> AgentResult:
        pending = state.pending_action
        if (
            pending is not None
            and logical_tool_name(pending.tool_name) == "create_ticket"
        ):
            if state.confirmation_status == ConfirmationStatus.PENDING:
                return AgentResult(
                    agent=self.name,
                    success=True,
                    response="请确认是否创建人工客服工单。",
                    pending_action=pending,
                    completed=False,
                )
            if state.confirmation_status == ConfirmationStatus.CONFIRMED:
                self._registry.confirm_action(pending.action_id)
                arguments = dict(pending.arguments)
                arguments["action_id"] = pending.action_id
                tool_result = await self._registry.call(
                    self.name,
                    pending.tool_name,
                    arguments,
                    context={"prompt_context": context} if context else None,
                    action_id=pending.action_id,
                )
                observation = tool_result.to_observation()
                if not tool_result.success:
                    return AgentResult(
                        agent=self.name,
                        success=False,
                        response="人工客服工单创建失败，请稍后重试。",
                        observations=[observation],
                        completed=False,
                        error=tool_result.error,
                    )
                ticket = TicketCreateOutput.model_validate(
                    tool_result.data or {},
                )
                if not ticket.created or not ticket.ticket_id:
                    return AgentResult(
                        agent=self.name,
                        success=True,
                        response=f"人工客服工单未能创建：{ticket.error or '业务系统未返回工单编号'}。",
                        observations=[observation],
                    )
                return AgentResult(
                    agent=self.name,
                    success=True,
                    response=f"人工客服工单已创建，工单编号：{ticket.ticket_id}。",
                    observations=[observation],
                )

        subject = (
            "用户投诉"
            if state.active_intent == "complaint"
            else "用户申请转人工"
        )
        action = PendingAction(
            tool_name=self.tool_name("create_ticket"),
            arguments={
                "subject": subject,
                "description": message or subject,
                **(
                    {"order_id": state.slots["order_id"]}
                    if state.slots.get("order_id") else {}
                ),
            },
        )
        self._registry.mark_pending_action(action.action_id)
        return AgentResult(
            agent=self.name,
            success=True,
            response="我可以为你创建人工客服工单。请确认是否创建。",
            pending_action=action,
            completed=False,
        )

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
            return self.tool_name("query_order"), {"order_id": order_id}, None
        if len(observations) == 1 and observations[0].data.get("found", True):
            return (
                self.tool_name("check_refund_eligibility"),
                {"order_id": order_id},
                None,
            )
        return None

    def finish(self, state, observations):
        pending = state.pending_action
        if (
            pending is not None
            and logical_tool_name(observations[-1].name) == "create_refund"
        ):
            refund = RefundCreateOutput.model_validate(
                observations[-1].data or {},
            )
            if not refund.created or not refund.refund_id:
                reason = refund.error or "业务系统未返回退款申请编号"
                return AgentResult(
                    agent=self.name,
                    success=True,
                    response=f"退款申请未能提交：{reason}。",
                    observations=observations,
                )
            return AgentResult(
                agent=self.name,
                success=True,
                response=f"退款申请已提交，申请编号：{refund.refund_id}。",
                observations=observations,
            )

        if logical_tool_name(observations[-1].name) == "query_order":
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
            tool_name=self.tool_name("create_refund"),
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

    def __init__(
        self,
        registry: ToolRegistry,
        query_planner: Optional[QueryPlanner] = None,
        answer_generator: Optional[RAGAnswerGenerator] = None,
    ) -> None:
        self._registry = registry
        self._query_planner = query_planner
        self._answer_generator = answer_generator
        canonical_name = canonical_tool_name("rag_search")
        self._rag_tool = (
            canonical_name
            if registry.get_spec(canonical_name) is not None
            else "rag_search"
        )
        self._registry.set_agent_whitelist(self.name, {self._rag_tool})

    @staticmethod
    def _contextual_query(message: str, context: str) -> str:
        """Add bounded policy and long-term context to ambiguous RAG queries."""
        if not context:
            return message
        sections = []
        for label in ("会话摘要", "相关历史", "用户画像"):
            marker = f"[{label}]\n"
            start = context.find(marker)
            if start < 0:
                continue
            start += len(marker)
            end = context.find("\n\n[", start)
            value = context[start:end if end >= 0 else None].strip()
            if value:
                sections.append(f"[{label}] {value[:500]}")
        if not sections:
            return message
        suffix = "\n".join(sections)
        return f"{message}\n\n检索上下文：\n{suffix}"[:1800]

    async def execute(
        self,
        state: DialogueState,
        message: str,
        context: str = "",
        history: Optional[List[Dict[str, str]]] = None,
        goal: Optional[str] = None,
    ) -> AgentResult:
        task_state = (
            state.model_copy(update={"active_intent": goal}, deep=True)
            if goal
            else state
        )
        deterministic = {
            "greeting": "你好，我可以帮你查询订单、物流，或处理退款相关问题。",
            "feedback": "感谢你的反馈，我已记录你的意见。",
            "other": "请说明你需要查询订单、物流，还是咨询退款政策。",
        }
        if task_state.active_intent in deterministic:
            return AgentResult(
                agent=self.name,
                success=True,
                response=deterministic[task_state.active_intent],
            )
        planner = self._query_planner or QueryPlanner(enabled=False)
        plan = await planner.plan(
            message,
            history=history,
            dialogue_state=task_state.model_dump(mode="json"),
        )
        primary_text = (
            plan.original_query
            if planner.max_queries == 1
            else plan.standalone_query
        )
        primary_query = self._contextual_query(
            primary_text,
            context,
        )
        queries = [primary_query]
        for query in [
            plan.original_query,
            *plan.alternative_queries,
        ]:
            if query not in queries:
                queries.append(query)
        queries = queries[:planner.max_queries]
        result = await self._registry.call(
            self.name,
            self._rag_tool,
            {
                "query": primary_query,
                "queries": queries,
                "mode": "rerank",
            },
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

        comparison = payload.get("temporal_comparison")
        if isinstance(comparison, dict):
            return AgentResult(
                agent=self.name,
                success=True,
                response=self._temporal_response(comparison),
                observations=[observation],
                citations=citations,
            )

        if self._answer_generator is not None:
            generated = await self._answer_generator.generate(
                message,
                primary_text,
                [item for item in items if isinstance(item, dict)],
                [item for item in citations if isinstance(item, dict)],
            )
            if generated.generated and not generated.sufficient_evidence:
                return AgentResult(
                    agent=self.name,
                    success=True,
                    response="根据现有资料无法回答该问题。",
                    observations=[observation],
                )
            if generated.sufficient_evidence:
                return AgentResult(
                    agent=self.name,
                    success=True,
                    response=generated.response,
                    observations=[observation],
                    citations=generated.citations,
                )
            logger.warning(
                "RAG answer generation failed, using extractive fallback: %s",
                generated.fallback_reason,
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

    @staticmethod
    def _temporal_response(comparison: Dict[str, Any]) -> str:
        current = comparison.get("current_version") or "当前版本"
        previous = comparison.get("previous_version")
        if not comparison.get("previous_found"):
            return (
                f"已找到当前知识版本 {current}，但没有检索到上一历史版本，"
                "暂时无法判断最近是否发生变化。[1]"
            )
        if not comparison.get("changed"):
            return (
                f"对比 {previous} 与 {current}，未发现政策正文变化。[1][2]"
            )
        parts = [f"对比 {previous} 与 {current}，政策存在以下变化："]
        added = comparison.get("added") or []
        removed = comparison.get("removed") or []
        if added:
            parts.append("新增：" + "；".join(str(item) for item in added[:3]))
        if removed:
            parts.append("删除：" + "；".join(str(item) for item in removed[:3]))
        parts.append("[1][2]")
        return "\n".join(parts)


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
        self._completion = GoalCompletionEvaluator()

    async def execute(
        self,
        state: DialogueState,
        message: str,
        intents: Optional[Sequence[str]] = None,
        context: str = "",
        history: Optional[List[Dict[str, str]]] = None,
    ) -> tuple[str, List[AgentResult]]:
        results: List[AgentResult] = []
        goals = list(dict.fromkeys(
            intents or [state.active_intent or "other"],
        ))
        tasks = [
            AgentTask(
                goal=goal,
                target_agent=self._router.route_intent(goal, state),
            )
            for goal in goals
        ]
        working_state = state
        shared_order_observations: Dict[str, Observation] = {}

        for task in tasks:
            missing_slots = get_missing_slots(task.goal, working_state.slots)
            if missing_slots:
                results.append(AgentResult(
                    agent=task.target_agent,
                    success=True,
                    response=(
                        "请提供订单号或物流单号。"
                        if task.goal == "logistics_query"
                        else f"请补充 {missing_slots[0]}。"
                    ),
                    completed=False,
                    goal=task.goal,
                    goals=[task.goal],
                    missing_slots=missing_slots,
                ))
                break

            agent = self._agents[task.target_agent]
            order_id = working_state.slots.get("order_id")
            seed = (
                [shared_order_observations[order_id]]
                if order_id in shared_order_observations
                and task.target_agent != KNOWLEDGE_AGENT
                else []
            )
            if task.target_agent == KNOWLEDGE_AGENT:
                result = await agent.execute(
                    working_state,
                    message,
                    context,
                    history=history,
                    goal=task.goal,
                )
            else:
                result = await agent.execute(
                    working_state,
                    message,
                    context=context,
                    goal=task.goal,
                    seed_observations=seed,
                )
            result.goal = task.goal
            result.goals = [task.goal]
            result.completed = self._completion.is_complete(
                task.goal,
                result,
            )
            results.append(result)
            for observation in result.observations:
                if (
                    logical_tool_name(observation.name) == "query_order"
                    and observation.success
                    and isinstance(observation.data, dict)
                    and observation.data.get("order_id")
                ):
                    shared_order_observations[
                        observation.data["order_id"]
                    ] = observation
            if result.pending_action is not None or not result.success:
                break
            if working_state.pending_action is not None:
                working_state = working_state.model_copy(update={
                    "pending_action": None,
                    "confirmation_status": ConfirmationStatus.NOT_REQUIRED,
                })
        return self._composer.compose(results), results
