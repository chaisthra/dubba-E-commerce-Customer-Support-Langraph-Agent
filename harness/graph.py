"""
The LangGraph harness. Same principle as harness/loop.py: the LLM proposes an
action; harness code decides whether it executes. Here that boundary is the
conditional edge between `propose` and `execute_tool` -- schema.validate_action()
runs inside propose_node (defensive shape check), and permissions.check_action_permission()
runs in route_after_propose(), an edge, not a node -- so ownership scoping is
architecturally a graph edge, not logic buried inside a tool call.

Nodes: classify -> propose -> (edge: permission check) -> execute_tool -> evaluate
-> (loop back to propose, or) respond / escalate. Categories are processed one at a
time, looping the whole chain per category (see route_after_category). A permission
denial short-circuits straight to reject_tool -- no retry within that category, ever.

tool_call_count and turn_tool_results are both scoped to the WHOLE turn (max 3 total
tool calls across every category; every tool result gathered stays visible to every
category, not just the one that fetched it) -- so if delivery_issue already looked
up "damaged item -> refund eligibility" via search_policy, refund_request (the same
underlying issue, processed right after) sees that result too instead of redundantly
re-discovering it against its own share of the shrinking cap. Neither resets between
categories, only at the start of a new turn (classify_node).
"""

from typing import TypedDict

from langfuse import get_client, propagate_attributes
from langgraph.graph import END, START, StateGraph

from harness import llm_client, permissions, prompts, schema
from harness.mcp_client import MCPClient
from memory.store import format_prior_tickets

REJECTED_ACTION_MESSAGE = (
    "Sorry, I hit an internal issue handling that part of your request. "
    "Could you rephrase it?"
)
PERMISSION_DENIED_MESSAGE = (
    "I can't access that -- it doesn't belong to your account, so I'm not able to "
    "look it up."
)
ESCALATE_MESSAGE = (
    "I wasn't able to fully resolve this after checking the information available to "
    "me, so I'm creating a support ticket for you now. One of our team will reach out "
    "shortly from dubba.support@dubba.com -- you'll get the full details by email."
)

TOOL_CALL_CAP = 3

_mcp_client: MCPClient | None = None


def set_mcp_client(client: MCPClient) -> None:
    global _mcp_client
    _mcp_client = client


class TicketState(TypedDict):
    session_id: str
    customer_id: str
    account: dict
    prior_tickets: list[dict]
    short_term_buffer: list[dict]
    user_message: str
    categories: list[str]
    category_index: int
    tool_call_count: int
    turn_tool_results: list[dict]
    proposed_action: dict | None
    final_actions: list[dict]
    sufficient: bool


CLASSIFY_TOOL = {
    "name": "classify",
    "description": "Return every ticket category the customer's message touches.",
    "input_schema": {
        "type": "object",
        "properties": {
            "categories": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(schema.VALID_CATEGORIES)},
                "minItems": 1,
            }
        },
        "required": ["categories"],
    },
}

PROPOSE_TOOLS = [
    {
        "name": "respond",
        "description": "Give a final answer for this category using only real information you already have.",
        "input_schema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    },
    {
        "name": "ask_clarification",
        "description": "Ask the customer something essential you're missing that a tool call can't resolve.",
        "input_schema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    },
    {
        "name": "lookup_order",
        "description": "Look up a single order's status, items, and delivery date.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "check_account_status",
        "description": "Look up an account's standing and order history.",
        "input_schema": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
        },
    },
    {
        "name": "search_policy",
        "description": (
            "Search Dubba's policy docs (refund eligibility, return eligibility, "
            "shipping delay compensation, pricing/shipping, account suspension "
            "appeals, subscription cancellation) for chunks relevant to a question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]

EVALUATE_TOOL = {
    "name": "evaluate",
    "description": "Judge whether the data gathered so far is sufficient to answer the customer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sufficient": {"type": "boolean"},
            "reasoning": {"type": "string"},
        },
        "required": ["sufficient", "reasoning"],
    },
}


def _advance(state: TicketState, category: str, message: str) -> dict:
    """Shared tail for respond/escalate/reject_tool: record this category's final
    message and move on. Does NOT touch tool_call_count or turn_tool_results -- both
    are shared across the WHOLE turn, not reset per category. If an earlier category
    already looked something up (e.g. delivery_issue searching policy for a damaged
    item), a later category (e.g. refund_request, same underlying issue) sees it
    too, instead of redundantly re-discovering it against its own shrinking budget."""
    return {
        "final_actions": state["final_actions"] + [{"category": category, "message": message}],
        "category_index": state["category_index"] + 1,
        "proposed_action": None,
    }


def classify_node(state: TicketState) -> dict:
    langfuse = get_client()
    with langfuse.start_as_current_observation(
        name="classify-intent", as_type="generation", model=llm_client.PRIMARY_MODEL
    ) as gen:
        gen.update(input=state["user_message"])
        response = llm_client.complete(
            system=prompts.CLASSIFY_SYSTEM_PROMPT,
            messages=state["short_term_buffer"],
            tools=[CLASSIFY_TOOL],
            force_tool="classify",
        )
        categories = response.tool_calls[0].input["categories"]
        gen.update(
            output=categories,
            model=response.model,
            usage_details={"input": response.input_tokens, "output": response.output_tokens},
        )

    return {
        "categories": categories,
        "category_index": 0,
        "tool_call_count": 0,
        "turn_tool_results": [],
        "final_actions": [],
    }


def propose_node(state: TicketState) -> dict:
    category = state["categories"][state["category_index"]]
    account = state["account"]
    system = (
        f"{prompts.PROPOSE_SYSTEM_PROMPT}\n\n"
        f"Category you are handling right now: {category}\n"
        f"Authenticated customer ID: {state['customer_id']}\n"
        f"This customer's order IDs, oldest first: {account['order_ids']}\n"
        f"Account standing: {account['standing']}\n"
        f"{format_prior_tickets(state['prior_tickets'])}\n"
        f"Tool results gathered so far THIS TURN, across all categories (reuse "
        f"anything relevant here instead of re-fetching it): {state['turn_tool_results']}"
    )

    langfuse = get_client()
    with langfuse.start_as_current_observation(
        name="propose-action", as_type="generation", model=llm_client.PRIMARY_MODEL
    ) as gen:
        gen.update(input=state["user_message"])
        response = llm_client.complete(
            system=system,
            messages=state["short_term_buffer"],
            tools=PROPOSE_TOOLS,
            any_tool=True,
        )
        tool_call = response.tool_calls[0]
        action = {"action_type": tool_call.name, "category": category, **tool_call.input}

        valid, reason = schema.validate_action(action)
        if not valid:
            # Defensive fallback: forced tool_choice should make this unreachable,
            # but never trust an unvalidated shape. Framed as ask_clarification so
            # respond_node skips a wasted extra LLM call and uses this text directly.
            action = {
                "action_type": "ask_clarification",
                "category": category,
                "message": REJECTED_ACTION_MESSAGE,
            }

        gen.update(
            output=action,
            model=response.model,
            usage_details={"input": response.input_tokens, "output": response.output_tokens},
        )

    return {"proposed_action": action}


def route_after_propose(state: TicketState) -> str:
    action = state["proposed_action"]
    action_type = action["action_type"]

    if action_type in schema.TOOL_ACTION_TYPES:
        allowed, _ = permissions.check_action_permission(action, state)
        return "execute_tool" if allowed else "reject_tool"

    return "respond"  # respond or ask_clarification both go through respond_node


async def execute_tool_node(state: TicketState) -> dict:
    action = state["proposed_action"]
    category = state["categories"][state["category_index"]]
    tool_name = action["action_type"]
    arg_field = schema.TOOL_ARG_FIELDS[tool_name]
    arguments = {arg_field: action[arg_field]}

    langfuse = get_client()
    with langfuse.start_as_current_observation(name=f"execute-tool:{tool_name}", as_type="tool") as span:
        span.update(input=arguments)
        result = await _mcp_client.call_tool(tool_name, arguments)
        span.update(output=result)

    return {
        "turn_tool_results": state["turn_tool_results"]
        + [{"tool": tool_name, "arguments": arguments, "result": result, "gathered_for_category": category}],
        "tool_call_count": state["tool_call_count"] + 1,
        "proposed_action": None,
    }


def evaluate_node(state: TicketState) -> dict:
    category = state["categories"][state["category_index"]]
    system = (
        f"{prompts.EVALUATE_SYSTEM_PROMPT}\n\n"
        f"Category: {category}\n"
        f"Gathered data so far, THIS TURN, across all categories: {state['turn_tool_results']}"
    )

    langfuse = get_client()
    with langfuse.start_as_current_observation(
        name="evaluate-sufficiency", as_type="generation", model=llm_client.PRIMARY_MODEL
    ) as gen:
        gen.update(input=state["user_message"])
        response = llm_client.complete(
            system=system,
            messages=[{"role": "user", "content": state["user_message"]}],
            tools=[EVALUATE_TOOL],
            force_tool="evaluate",
        )
        judgment = response.tool_calls[0].input
        gen.update(
            output=judgment,
            model=response.model,
            usage_details={"input": response.input_tokens, "output": response.output_tokens},
        )

    return {"sufficient": judgment["sufficient"]}


def route_after_evaluate(state: TicketState) -> str:
    if state["sufficient"]:
        return "respond"
    if state["tool_call_count"] >= TOOL_CALL_CAP:
        return "escalate"
    return "propose"


def respond_node(state: TicketState) -> dict:
    category = state["categories"][state["category_index"]]
    action = state["proposed_action"]
    # action is None when respond is reached via evaluate -> respond (proposed_action
    # was already cleared by execute_tool_node after the last tool call) -- that path
    # always needs the LLM formatting call below, same as a fresh "respond" proposal.
    # Only a direct propose -> respond with ask_clarification skips it.
    if action is not None and action["action_type"] == "ask_clarification":
        message = action["message"]
    else:
        system = (
            f"{prompts.RESPOND_SYSTEM_PROMPT}\n\n"
            f"Category: {category}\n"
            f"Gathered data, THIS TURN, across all categories: {state['turn_tool_results']}"
        )
        langfuse = get_client()
        with langfuse.start_as_current_observation(
            name="respond-format", as_type="generation", model=llm_client.PRIMARY_MODEL
        ) as gen:
            gen.update(input=state["user_message"])
            response = llm_client.complete(
                system=system,
                messages=state["short_term_buffer"],
            )
            message = response.text
            gen.update(
                output=message,
                model=response.model,
                usage_details={"input": response.input_tokens, "output": response.output_tokens},
            )

    return _advance(state, category, message)


def escalate_node(state: TicketState) -> dict:
    category = state["categories"][state["category_index"]]
    langfuse = get_client()
    with langfuse.start_as_current_observation(name="escalate-hitl", as_type="span") as span:
        span.update(
            input={"category": category, "tool_call_count": state["tool_call_count"]},
            output=ESCALATE_MESSAGE,
            level="WARNING",
        )
    return _advance(state, category, ESCALATE_MESSAGE)


def reject_tool_node(state: TicketState) -> dict:
    category = state["categories"][state["category_index"]]
    action = state["proposed_action"]
    _, reason = permissions.check_action_permission(action, state)  # edge already gated; this call is for the log entry only

    langfuse = get_client()
    with langfuse.start_as_current_observation(name="reject-tool-call", as_type="span") as span:
        span.update(input=action, output={"allowed": False, "reason": reason}, level="ERROR")

    return _advance(state, category, PERMISSION_DENIED_MESSAGE)


def route_after_category(state: TicketState) -> str:
    if state["category_index"] < len(state["categories"]):
        return "propose"
    return END


def build_graph():
    workflow = StateGraph(TicketState)
    workflow.add_node("classify", classify_node)
    workflow.add_node("propose", propose_node)
    workflow.add_node("execute_tool", execute_tool_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("respond", respond_node)
    workflow.add_node("escalate", escalate_node)
    workflow.add_node("reject_tool", reject_tool_node)

    workflow.add_edge(START, "classify")
    workflow.add_edge("classify", "propose")
    workflow.add_conditional_edges(
        "propose",
        route_after_propose,
        {"execute_tool": "execute_tool", "respond": "respond", "reject_tool": "reject_tool"},
    )
    workflow.add_edge("execute_tool", "evaluate")
    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"respond": "respond", "propose": "propose", "escalate": "escalate"},
    )
    workflow.add_conditional_edges("respond", route_after_category, {"propose": "propose", END: END})
    workflow.add_conditional_edges("escalate", route_after_category, {"propose": "propose", END: END})
    workflow.add_conditional_edges("reject_tool", route_after_category, {"propose": "propose", END: END})

    return workflow.compile()


_compiled_graph = build_graph()


def _compose(actions: list[dict]) -> str:
    if len(actions) == 1:
        return actions[0]["message"]
    parts = []
    for action in actions:
        label = action["category"].replace("_", " ").title()
        parts.append(f"[{label}]\n{action['message']}")
    return "\n\n".join(parts)


async def run_turn(session: dict, user_message: str) -> str:
    langfuse = get_client()
    session["short_term_buffer"].append({"role": "user", "content": user_message})

    with langfuse.start_as_current_observation(name="ticket-turn", as_type="span") as trace:
        trace.update(input=user_message)
        with propagate_attributes(user_id=session["customer_id"], session_id=session["session_id"]):
            initial_state: TicketState = {
                "session_id": session["session_id"],
                "customer_id": session["customer_id"],
                "account": session["account"],
                "prior_tickets": session["prior_tickets"],
                "short_term_buffer": session["short_term_buffer"],
                "user_message": user_message,
                "categories": [],
                "category_index": 0,
                "tool_call_count": 0,
                "turn_tool_results": [],
                "proposed_action": None,
                "final_actions": [],
                "sufficient": False,
            }
            final_state = await _compiled_graph.ainvoke(initial_state)
            trace.update(metadata={"categories": ",".join(final_state["categories"])})
            reply = _compose(final_state["final_actions"])
        trace.update(output=reply)

    session["short_term_buffer"].append({"role": "assistant", "content": reply})
    langfuse.flush()
    return reply
