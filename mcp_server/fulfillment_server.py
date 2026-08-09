"""Fulfillment MCP server for delivery tracking and diagnosis."""

from mcp.server.fastmcp import FastMCP

from mcp_server import customer_service_server as backend


server = FastMCP(
    "ggbot-fulfillment",
    instructions="Package tracking, delivery estimate, and exception tools.",
)

for tool in (
    backend.track_package,
    backend.estimate_delivery,
    backend.diagnose_delivery_exception,
):
    server.tool()(tool)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
