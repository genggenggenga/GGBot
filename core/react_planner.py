"""Bounded structured ReAct planning for domain agents."""
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Dict, Sequence

from core.agent_models import AgentDecision, DialogueState, Observation
from core.tool_registry import ToolSpec


LLMCall = Callable[[str], Awaitable[str]]


class ReActPlanner:
    """Ask an LLM for one structured action at a time."""

    def __init__(self, llm_call: LLMCall, *, max_observation_chars: int = 2400):
        self._llm_call = llm_call
        self._max_observation_chars = max_observation_chars

    async def decide(
        self,
        *,
        agent_name: str,
        goal: str,
        message: str,
        state: DialogueState,
        observations: Sequence[Observation],
        tools: Sequence[ToolSpec],
        system_prompt: str = "",
    ) -> AgentDecision:
        raw = await self._llm_call(self._build_prompt(
            agent_name=agent_name,
            goal=goal,
            message=message,
            state=state,
            observations=observations,
            tools=tools,
            system_prompt=system_prompt,
        ))
        return AgentDecision.model_validate(self._parse_json(raw))

    def _build_prompt(
        self,
        *,
        agent_name: str,
        goal: str,
        message: str,
        state: DialogueState,
        observations: Sequence[Observation],
        tools: Sequence[ToolSpec],
        system_prompt: str,
    ) -> str:
        tool_contracts = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "tool_type": tool.tool_type.value,
            }
            for tool in tools
        ]
        observation_data = [
            {
                "name": item.name,
                "success": item.success,
                "data": item.data,
                "error": item.error,
            }
            for item in observations
        ]
        observation_text = json.dumps(
            observation_data,
            ensure_ascii=False,
            default=str,
        )[:self._max_observation_chars]
        state_data: Dict[str, Any] = {
            "active_intent": state.active_intent,
            "slots": state.slots,
            "confirmation_status": state.confirmation_status.value,
            "completed_goals": state.completed_goals,
        }
        return f"""你是 {agent_name} 领域 Agent 的受约束决策器。
{system_prompt}

目标：{goal}
用户消息：{message}
业务状态：{json.dumps(state_data, ensure_ascii=False, default=str)}
已获得的 Observation：{observation_text}
允许使用的工具：{json.dumps(tool_contracts, ensure_ascii=False)}

每次只决定一个下一步，严格遵守：
1. 只能选择允许列表中的工具，不得虚构工具或参数。
2. 工具参数只能来自用户消息、业务状态或 Observation。
3. 已有足够事实时返回 finish；缺少必要信息时返回 clarify。
4. 不要重复调用相同工具和相同参数。
5. 写工具只能提出调用建议，执行层会负责用户确认。
6. 最终回答只能基于业务状态和 Observation，不得编造事实。
7. 只输出 JSON，不输出 Thought 或其他文字。

输出格式：
{{
  "type": "tool|finish|clarify|handoff",
  "tool_name": "工具名或 null",
  "arguments": {{}},
  "response": "finish/clarify/handoff 时必填",
  "reason_code": "简短稳定的原因编码"
}}"""

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        text = str(raw).strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("ReAct planner output is not JSON")
        data = json.loads(text[start:end])
        if not isinstance(data, dict):
            raise ValueError("ReAct planner output must be an object")
        return data
