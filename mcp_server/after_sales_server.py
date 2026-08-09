"""After-sales MCP server for eligibility checks and confirmed writes."""

from mcp.server.fastmcp import FastMCP

from mcp_server import customer_service_server as backend


server = FastMCP(
    "ggbot-after-sales",
    instructions="Refund, return, cancellation, and handoff tools.",
)

for tool in (
    backend.check_refund_eligibility,
    backend.evaluate_after_sales_options,
    backend.calculate_refund_quote,
    backend.create_refund,
    backend.create_return,
    backend.cancel_order,
    backend.create_ticket,
):
    server.tool()(tool)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
