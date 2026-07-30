"""
The permission-scoping check. One clear function.

Every action the LLM proposes passes through check_action_permission() before the
harness executes anything -- called AFTER schema.validate_action() has already
confirmed the action is well-formed. This file only answers business-rule questions
("is this action allowed to run"), never shape questions ("is this action well-formed")
-- that split keeps each check testable on its own.

Ownership checks (does this order/customer id belong to session["customer_id"]) live
here as their own functions, called by harness/graph.py's conditional edge between
propose and execute_tool -- never inside mcp_server/server.py's tool functions
themselves. See log/WHY.md for why that boundary matters.
"""

from mcp_server.mock_data import get_order

# Action types the harness is currently willing to execute at all.
ALLOWED_ACTION_TYPES = {
    "respond",
    "ask_clarification",
    "lookup_order",
    "check_account_status",
    "search_policy",
}


def check_order_permission(order_id: str, session: dict) -> tuple[bool, str]:
    """Returns (allowed, reason). Denies if the order doesn't exist, or exists but
    belongs to a different customer than the authenticated session."""
    order = get_order(order_id)
    if order is None:
        return False, f"order_id={order_id!r} does not exist"
    if order["customer_id"] != session["customer_id"]:
        return False, (
            f"order_id={order_id!r} belongs to a different customer "
            f"than the authenticated session ({session['customer_id']!r})"
        )
    return True, "order belongs to the authenticated customer"


def check_account_permission(customer_id: str, session: dict) -> tuple[bool, str]:
    """Returns (allowed, reason). Denies any check_account_status call for a
    customer_id other than the session's own -- even though a well-behaved model
    would only ever ask about itself, this must never exist in an unscoped form."""
    if customer_id != session["customer_id"]:
        return False, (
            f"customer_id={customer_id!r} does not match the authenticated "
            f"session ({session['customer_id']!r})"
        )
    return True, "customer_id matches the authenticated session"


def check_action_permission(action: dict, session: dict) -> tuple[bool, str]:
    """Returns (allowed, reason). `reason` is always populated, even on
    success, so every decision is traceable. Assumes `action` already passed
    schema.validate_action()."""
    action_type = action["action_type"]

    if action_type not in ALLOWED_ACTION_TYPES:
        return False, (
            f"action_type={action_type!r} is not enabled in this build "
            f"(allowed: {sorted(ALLOWED_ACTION_TYPES)})"
        )

    if action_type == "lookup_order":
        return check_order_permission(action["order_id"], session)

    if action_type == "check_account_status":
        return check_account_permission(action["customer_id"], session)

    if action_type in ("respond", "ask_clarification", "search_policy"):
        return True, f"{action_type} is always permitted (no order/customer id involved)"

    return False, f"unhandled action_type={action_type!r}"
