"""
Shape/type validation for the LLM's proposed action -- distinct from permission
scoping (permissions.py), which is a business-rule check, not a shape check.

Claude's forced tool_choice already constrains the response shape at the API level;
this is a defensive second check so the harness never trusts an unvalidated dict,
and so there is one exact function to point at for "malformed action gets rejected."

Shared by both harness implementations (harness/loop.py, harness/graph.py) -- the
while-loop only ever produces respond/ask_clarification actions; the graph also
produces lookup_order/check_account_status when it proposes a real MCP tool call.
"""

TOOL_ARG_FIELDS = {
    "lookup_order": "order_id",
    "check_account_status": "customer_id",
    "search_policy": "query",
}
TOOL_ACTION_TYPES = set(TOOL_ARG_FIELDS)
# search_policy has no customer/order ID to own -- it searches public policy docs,
# not customer data -- so it skips the ownership check in permissions.py entirely.
UNSCOPED_TOOL_ACTION_TYPES = {"search_policy"}

VALID_ACTION_TYPES = {"respond", "ask_clarification"} | TOOL_ACTION_TYPES
VALID_CATEGORIES = {
    "order_status",
    "delivery_issue",
    "refund_request",
    "subscription_account",
    "other",
}


def validate_action(action: dict) -> tuple[bool, str]:
    """Returns (valid, reason). `reason` is always populated."""
    if not isinstance(action, dict):
        return False, f"action must be a dict, got {type(action).__name__}"

    action_type = action.get("action_type")
    if action_type not in VALID_ACTION_TYPES:
        return False, f"action_type must be one of {sorted(VALID_ACTION_TYPES)}, got {action_type!r}"

    category = action.get("category")
    if category not in VALID_CATEGORIES:
        return False, f"category must be one of {sorted(VALID_CATEGORIES)}, got {category!r}"

    if action_type in TOOL_ACTION_TYPES:
        arg_field = TOOL_ARG_FIELDS[action_type]
        arg_value = action.get(arg_field)
        if not isinstance(arg_value, str) or not arg_value.strip():
            return False, f"{action_type} requires a non-empty string {arg_field!r}, got {arg_value!r}"
        return True, f"{action_type} action shape is valid"

    message = action.get("message")
    if not isinstance(message, str) or not message.strip():
        return False, "message must be a non-empty string"

    return True, "action shape is valid"
