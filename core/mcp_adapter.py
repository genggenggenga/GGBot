"""Official MCP stdio client and ToolRegistry adapter."""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from core.tool_registry import (
    CircuitBreaker,
    ToolResult,
    ToolSpec,
    ToolStats,
    ToolType,
    validate_params,
)


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


@dataclass(frozen=True)
class NamespacedMCPTool:
    """MCP tool metadata with the registry namespace applied."""

    name: str
    title: Optional[str]
    description: str
    inputSchema: Dict[str, Any]
    outputSchema: Optional[Dict[str, Any]]
    annotations: Any = None


class MCPClientManager:
    """Own all domain MCP clients and expose a combined tool catalogue."""

    def __init__(self, clients: Dict[str, MCPClient]) -> None:
        self._clients = dict(clients)

    @property
    def clients(self) -> Dict[str, MCPClient]:
        return dict(self._clients)

    async def connect(self) -> None:
        connected: List[MCPClient] = []
        try:
            for client in self._clients.values():
                await client.connect()
                connected.append(client)
        except Exception:
            for client in reversed(connected):
                await client.close()
            raise

    async def close(self) -> None:
        for client in reversed(list(self._clients.values())):
            try:
                await client.close()
            except Exception:
                # Continue closing the remaining subprocess sessions.
                pass

    async def list_tools(self) -> List[NamespacedMCPTool]:
        tools: List[NamespacedMCPTool] = []
        for namespace, client in self._clients.items():
            for tool in await client.list_tools():
                tools.append(NamespacedMCPTool(
                    name=f"{namespace}.{tool.name}",
                    title=getattr(tool, "title", None),
                    description=getattr(tool, "description", None) or "",
                    inputSchema=getattr(tool, "inputSchema", None)
                    or {"type": "object"},
                    outputSchema=getattr(tool, "outputSchema", None),
                    annotations=getattr(tool, "annotations", None),
                ))
        return tools


class MCPToolAdapter:
    """Adapts one discovered MCP tool to the ToolRegistry protocol."""

    def __init__(
        self,
        client: MCPClient,
        spec: ToolSpec,
        *,
        remote_name: Optional[str] = None,
    ) -> None:
        self._client = client
        self._remote_name = remote_name or spec.name
        self.spec = spec
        self.stats = ToolStats()
        self.breaker = CircuitBreaker()

    async def call(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
    ) -> ToolResult:
        del context, use_cache
        started = time.monotonic()
        self.stats.total += 1
        if not self.breaker.allow():
            self.stats.failed += 1
            self.stats.consecutive_fails += 1
            return ToolResult(
                success=False,
                tool_name=self.spec.name,
                error=f"tool circuit open: {self.spec.name}",
            )
        try:
            validate_params(self.spec, params)
            data = await asyncio.wait_for(
                self._client.call_tool(self._remote_name, params),
                timeout=self.spec.timeout_s,
            )
            latency_ms = (time.monotonic() - started) * 1000
            self.stats.success += 1
            self.stats.consecutive_fails = 0
            self.stats.total_latency_ms += latency_ms
            self.breaker.record_success()
            return ToolResult(
                success=True,
                data=data,
                tool_name=self.spec.name,
                latency_ms=round(latency_ms, 2),
            )
        except asyncio.TimeoutError:
            self.stats.failed += 1
            self.stats.consecutive_fails += 1
            self.breaker.record_failure()
            return ToolResult(
                success=False,
                tool_name=self.spec.name,
                error="execution timeout",
                latency_ms=round((time.monotonic() - started) * 1000, 2),
            )
        except Exception as ex:
            self.stats.failed += 1
            self.stats.consecutive_fails += 1
            self.breaker.record_failure()
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
        namespace: Optional[str] = None,
        write_tools: Optional[set[str]] = None,
        timeout_s: float = 30.0,
    ) -> List["MCPToolAdapter"]:
        write_tools = write_tools or set()
        adapters: List[MCPToolAdapter] = []
        for tool in await client.list_tools():
            registered_name = (
                f"{namespace}.{tool.name}"
                if namespace
                else tool.name
            )
            spec = ToolSpec(
                name=registered_name,
                description=tool.description or tool.name,
                input_schema=tool.inputSchema or {"type": "object"},
                tool_type=(
                    ToolType.WRITE if tool.name in write_tools else ToolType.READ
                ),
                timeout_s=timeout_s,
            )
            adapters.append(cls(client, spec, remote_name=tool.name))
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
