"""
FastMCP-style MCP server, stdio transport. Run as a subprocess of the agent process
(launched by harness/mcp_client.py), not as a standalone service.

Uses mcp.server.mcpserver.MCPServer -- the current SDK's name for the "FastMCP"
pattern the assignment describes (decorator-based tool registration with
type-hint-derived JSON schema validation); verified against the installed package
since the SDK renamed FastMCP -> MCPServer after the assignment was written.

Schema validation happens here, for free, from the type hints below -- a malformed
call (wrong type, missing field) is rejected by the tool layer itself. Permission
scoping is NOT here: see harness/permissions.py. The harness checks ownership BEFORE
ever calling these tools; these functions have no concept of "whose session this is."
"""

from mcp.server.mcpserver import MCPServer

from mcp_server.mock_data import get_account, get_order
from rag.retriever import retrieve

server = MCPServer("dubba-order-support")


@server.tool()
def lookup_order(order_id: str) -> dict:
    """Look up a single order's status, items, and delivery date by order ID."""
    order = get_order(order_id)
    if order is None:
        return {"error": f"no order found with order_id={order_id!r}"}
    return order


@server.tool()
def check_account_status(customer_id: str) -> dict:
    """Look up an account's standing and order history by customer ID."""
    account = get_account(customer_id)
    if account is None:
        return {"error": f"no account found with customer_id={customer_id!r}"}
    # Deliberately omit email/access_code -- tool responses shouldn't carry auth material.
    return {
        "customer_id": account["customer_id"],
        "standing": account["standing"],
        "order_ids": account["order_ids"],
    }


@server.tool()
def search_policy(query: str) -> dict:
    """Search Dubba's policy docs (refunds, returns, shipping delays, pricing,
    account suspension, subscriptions) and return the top matching chunks."""
    chunks = retrieve(query)
    return {"chunks": chunks}


if __name__ == "__main__":
    server.run()
