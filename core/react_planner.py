"""Bounded structured ReAct planning for domain agents."""
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Dict, Sequence

from core.agent_models import AgentDecision, DialogueState, Observation
from core.prompts.react import build_prompt as build_react_prompt
from core.prompts.types import PromptSpec
from core.tool_registry import ToolSpec


LLMCall = Callable[[PromptSpec], Awaitable[str]]


class ReActPlanner:
    """Ask an LLM for one structured action at a time."""

    def __init__(
        self,
        llm_call: LLMCall,
        *,
        structured_client: Any = None,
        max_observation_chars: int = 2400,
    ):
        self._llm_call = llm_call
        self._structured_client = structured_client
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
        skill_context: str = "",
    ) -> AgentDecision:
        prompt = self._build_prompt(
            agent_name=agent_name,
            goal=goal,
            message=message,
            state=state,
            observations=observations,
            tools=tools,
            system_prompt=system_prompt,
            skill_context=skill_context,
        )
        if self._structured_client is not None:
            return await self._structured_client.generate(
                prompt,
                AgentDecision,
                tool_name="submit_agent_decision",
                max_tokens=512,
                temperature=0.0,
            )
        raw = await self._llm_call(prompt)
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
        skill_context: str,
    ) -> PromptSpec:
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
        domain_policy = system_prompt
        if skill_context:
            domain_policy = (
                f"{system_prompt}\n\n[售后 Skill 软策略]\n{skill_context}"
            )
        return build_react_prompt(
            agent_name=agent_name,
            domain_policy=domain_policy,
            goal=goal,
            message=message,
            state=state_data,
            observations=observation_text,
            tools=tool_contracts,
        )

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        text = str(raw).strip()
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("ReAct planner output must be an object")
        return data
