"""Commerce MCP server for order and payment read tools."""

from mcp.server.fastmcp import FastMCP

from mcp_server import customer_service_server as backend


server = FastMCP(
    "ggbot-commerce",
    instructions="Order, item, payment, and invoice query tools.",
)

for tool in (
    backend.query_order,
    backend.query_order_items,
    backend.query_payment_detail,
    backend.query_invoice,
):
    server.tool()(tool)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
