"""Provider-native structured output using Anthropic tool calling."""
from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from core.prompts.types import PromptSpec


ModelT = TypeVar("ModelT", bound=BaseModel)


class StructuredOutputError(ValueError):
    """The provider did not return the required structured tool input."""


class StructuredLLMClient:
    """Generate Pydantic models through a forced provider tool call."""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        disable_thinking: bool = False,
    ) -> None:
        self._client = client
        self._model = model
        self._disable_thinking = disable_thinking

    async def generate(
        self,
        prompt: PromptSpec,
        output_model: type[ModelT],
        *,
        tool_name: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ModelT:
        # DeepSeek v4 系列默认开启 thinking 模式，与强制 tool_choice 不兼容
        # （返回 400: "Thinking mode does not support this tool_choice"）。
        # 第三方兼容端点需显式禁用 thinking 以解锁结构化工具调用。
        # anthropic SDK 0.40 未暴露 thinking 关键字参数，通过 extra_body 注入。
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": prompt.system,
            "messages": [{"role": "user", "content": prompt.user}],
            "tools": [{
                "name": tool_name,
                "description": "Return the required structured result.",
                "input_schema": output_model.model_json_schema(),
            }],
            "tool_choice": {"type": "tool", "name": tool_name},
        }
        if self._disable_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        response = await self._client.messages.create(**kwargs)
        payload = _tool_input(response.content, tool_name)
        return output_model.model_validate(payload)


def _tool_input(content: Any, tool_name: str) -> Any:
    for block in content or []:
        block_type = getattr(block, "type", None)
        name = getattr(block, "name", None)
        value = getattr(block, "input", None)
        if isinstance(block, dict):
            block_type = block.get("type", block_type)
            name = block.get("name", name)
            value = block.get("input", value)
        if block_type == "tool_use" and name == tool_name:
            return value
    raise StructuredOutputError(
        f"provider did not call required tool: {tool_name}",
    )
