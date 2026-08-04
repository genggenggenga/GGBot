"""Official MCP stdio client and ToolRegistry adapter."""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from core.tool_registry import ToolResult, ToolSpec, ToolType, validate_params


class MCPClient:
    """Owns one initialized MCP stdio session."""

    def __init__(
        self,
        command: str,
        args: List[str],
        *,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self._parameters = StdioServerParameters(command=command, args=args, env=env)
        self._stack: Optional[AsyncExitStack] = None
        self._session: Optional[ClientSession] = None

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("MCP client is not connected")
        return self._session

    async def connect(self) -> None:
        if self._stack is not None:
            return
        stack = AsyncExitStack()
        try:
            read_stream, write_stream = await stack.enter_async_context(
                stdio_client(self._parameters)
            )
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
        except Exception:
            await stack.aclose()
            raise
        self._stack = stack
        self._session = session

    async def close(self) -> None:
        stack, self._stack = self._stack, None
        self._session = None
        if stack is not None:
            await stack.aclose()

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()

    async def list_tools(self) -> List[Any]:
        result = await self.session.list_tools()
        return list(result.tools)

    async def call_tool(self, name: str, params: Dict[str, Any]) -> Any:
        result = await self.session.call_tool(name, arguments=params)
        if result.isError:
            message = _content_text(result.content) or f"MCP tool failed: {name}"
            raise RuntimeError(message)
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            if set(structured) == {"result"}:
                return structured["result"]
            return structured
        text = _content_text(result.content)
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text


class MCPToolAdapter:
    """Adapts one discovered MCP tool to the ToolRegistry protocol."""

    def __init__(self, client: MCPClient, spec: ToolSpec) -> None:
        self._client = client
        self.spec = spec

    async def call(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
    ) -> ToolResult:
        del context, use_cache
        started = time.monotonic()
        try:
            validate_params(self.spec, params)
            data = await asyncio.wait_for(
                self._client.call_tool(self.spec.name, params),
                timeout=self.spec.timeout_s,
            )
            return ToolResult(
                success=True,
                data=data,
                tool_name=self.spec.name,
                latency_ms=round((time.monotonic() - started) * 1000, 2),
            )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                tool_name=self.spec.name,
                error="execution timeout",
                latency_ms=round((time.monotonic() - started) * 1000, 2),
            )
        except Exception as ex:
            return ToolResult(
                success=False,
                tool_name=self.spec.name,
                error=str(ex),
                latency_ms=round((time.monotonic() - started) * 1000, 2),
            )

    @classmethod
    async def discover(
        cls,
        client: MCPClient,
        *,
        write_tools: Optional[set[str]] = None,
        timeout_s: float = 30.0,
    ) -> List["MCPToolAdapter"]:
        write_tools = write_tools or set()
        adapters: List[MCPToolAdapter] = []
        for tool in await client.list_tools():
            spec = ToolSpec(
                name=tool.name,
                description=tool.description or tool.name,
                input_schema=tool.inputSchema or {"type": "object"},
                tool_type=(
                    ToolType.WRITE if tool.name in write_tools else ToolType.READ
                ),
                timeout_s=timeout_s,
            )
            adapters.append(cls(client, spec))
        return adapters


def _content_text(content: List[Any]) -> str:
    texts = []
    for block in content or []:
        text = getattr(block, "text", None)
        if isinstance(block, dict):
            text = block.get("text", text)
        if isinstance(text, str):
            texts.append(text)
    return "\n".join(texts)
