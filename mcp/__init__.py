"""Compatibility package for legacy GGBot modules and the official MCP SDK.

GGBot historically used ``mcp`` as a local package name.  Extending the
package path keeps ``mcp.tool_manager`` importable while allowing official SDK
submodules such as ``mcp.client`` and ``mcp.server`` to resolve normally.
"""
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

# Preserve the official SDK's common top-level imports despite the local
# compatibility package taking precedence on sys.path.
from .client.session import ClientSession
from .client.stdio import StdioServerParameters, stdio_client
from .server.session import ServerSession
from .server.stdio import stdio_server
from .shared.exceptions import McpError

__all__ = [
    "ClientSession",
    "McpError",
    "ServerSession",
    "StdioServerParameters",
    "stdio_client",
    "stdio_server",
]
