"""
The LangGraph harness. Same principle as harness/loop.py: the LLM proposes an
action; harness code decides whether it executes. Here that boundary is the
conditional edge between `propose` and `execute_tool` -- schema.validate_action()
runs inside propose_node (defensive shape check), and permissions.check_action_permission()
runs in route_after_propose(), an edge, not a node -- so ownership scoping is
architecturally a graph edge, not logic buried inside a tool call.

Nodes: ingest_message -> (edge: token-cap check) -> classify -> propose -> (edge:
permission check) -> execute_tool -> evaluate -> (loop back to propose, or) respond /
escalate. Categories are processed one at a time, looping the whole chain per category
(see route_after_category). A permission denial short-circuits straight to
reject_tool -- no retry within that category, ever. Every path converges on
finalize_turn before END -- that's the one place a reply gets composed, appended to
short-term memory, and (periodically) rolled into the XML summary.

tool_call_count and turn_tool_results are both scoped to the WHOLE turn (max 3 total
tool calls across every category; every tool result gathered stays visible to every
category, not just the one that fetched it) -- so if delivery_issue already looked
up "damaged item -> refund eligibility" via search_policy, refund_request (the same
underlying issue, processed right after) sees that result too instead of redundantly
re-discovering it against its own share of the shrinking cap. Neither resets between
categories, only at the start of a new turn (classify_node).

respond does NOT write customer-facing prose per category. It only decides, per
category, whether that category is answerable (deferred to pending_response_categories)
or needs a clarifying question (written immediately -- short, category-specific,
low duplication risk). Once every category has been through the loop, finalize runs
ONE LLM call covering every pending category together -- this is what stops two
categories about the same underlying issue (e.g. delivery_issue + refund_request for
one damaged candle) from each independently writing near-identical prose. All
evaluate/tool-call work for every category finishes before any prose is written, not
interleaved category-by-category.

Session-scoped state (session_id, customer_id, account, prior_tickets,
short_term_buffer, conversation_summary_xml, turn_count, session_tool_log,
session_permission_denials) is owned by the Postgres
checkpointer (memory/checkpointer.py), keyed by thread_id = session_id -- run_turn()
seeds it once on a session's first turn and passes only {"user_message": ...} on every
turn after that, relying on the checkpoint to carry the rest forward. Turn-scoped
fields (categories, category_index, tool_call_count, turn_tool_results,
proposed_action, pending_response_categories, final_actions, sufficient, reply) live
in the same TypedDict for simplicity, but every one of them is unconditionally
overwritten by ingest_message/classify_node/finalize_turn before anything downstream
reads it, every single turn -- so checkpointing them alongside the session-scoped
fields never leaks stale turn data across turns, only across categories within one
turn (which is intentional, see above).

conversation_summary_xml never enters the `messages` list sent to the Anthropic API
(that list must stay a strict user/assistant alternation) -- it's injected into system
prompts instead, the same way account/prior_tickets already are.
"""

import os
from typing import TypedDict

from langfuse import get_client, propagate_attributes
from langgraph.graph import END, START, StateGraph

from harness import llm_client, permissions, prompts, schema
from harness.llm_provider import get_llm_client
from harness.mcp_client import MCPClient
from harness.summarizer import derive_order_id, summarize_turns
from memory.store import format_prior_tickets, relevant_prior_summary

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
SESSION_LIMIT_MESSAGE = (
    "This conversation has grown longer than I can safely keep track of in one go, "
    "so I'm flagging it for a teammate to pick up directly -- they'll have the full "
    "context and will follow up shortly rather than risk missing something."
)

TOOL_CALL_CAP = 3

# Rolling short-term-memory summarization (log/SESSION_DESIGN.md). TOKEN_CAP is the
# hard safety net -- if the buffer somehow blows past it before a periodic pass fires
# (SUMMARIZE_EVERY_N_TURNS), that turn skips straight to HITL escalation instead of
# risking an ever-growing, never-summarized session.
TOKEN_CAP = 8000
SUMMARIZE_EVERY_N_TURNS = 5
RAW_TURNS_KEPT_AFTER_SUMMARY = 4  # last 2 user+assistant exchanges stay verbatim

_mcp_client: MCPClient | None = None
_compiled_graph = None


def set_mcp_client(client: MCPClient) -> None:
    global _mcp_client
    _mcp_client = client


def init_graph(checkpointer) -> None:
    """Must be called once at app startup, after the Postgres checkpointer is ready
    (see memory/checkpointer.py + main.py) -- the graph can't compile with a
    checkpointer that doesn't exist yet, so this replaces the old eager
    module-level `_compiled_graph = build_graph()`."""
    global _compiled_graph
    _compiled_graph = build_graph(checkpointer)


class TicketState(TypedDict):
    session_id: str
    customer_id: str
    account: dict
    prior_tickets: list[dict]
    short_term_buffer: list[dict]
    conversation_summary_xml: str
    turn_count: int
    user_message: str
    categories: list[str]
    category_index: int
    tool_call_count: int
    turn_tool_results: list[dict]
    proposed_action: dict | None
    pending_response_categories: list[str]
    final_actions: list[dict]
    sufficient: bool
    cap_breached: bool
    reply: str
    # Session-scoped, appended-to-never-reset (unlike turn_tool_results/tool_call_count,
    # which reset every turn in classify_node) -- the harness's own durable record of
    # what happened this session, so summarize_turns() can build tool_calls_made/
    # tool_results/permission_checks/order_id deterministically instead of asking a
    # model to reconstruct them from prose. See log/DECISIONS.md.
    session_tool_log: list[dict]
    session_permission_denials: list[dict]


CLASSIFY_TOOL = {
    "name": "classify",
    "description": "Return every ticket category the customer's message touches.",
    # strict=true + additionalProperties=false -- see the identical comment on
    # _ALL_PROPOSE_TOOLS below for why every tool definition in this file has
    # this now, not just the ones directly implicated in the incident that
    # found the gap.
    "strict": True,
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
        "additionalProperties": False,
    },
}

# Assignment 2, section 2.4's deliberate-regression demo: read once at import time
# (not per-call) so a single env var flips behavior for a whole CI run without any
# code branch. When true, lookup_order is entirely absent from the tools list the
# LLM API call is given -- not just discouraged in prose, structurally impossible
# for the model to select (tool_choice enumerates only the tools actually passed
# in that call). Every ticket needing an order lookup then fails the trajectory
# check, demonstrating the gate blocking a real regression. See
# .github/workflows/ci-cd.yml and README.md for how this gets toggled in CI.
AGENT_REGRESSED = os.environ.get("AGENT_REGRESSED", "").lower() == "true"

# strict=true + additionalProperties=false on every tool below (and on
# CLASSIFY_TOOL/EVALUATE_TOOL) -- 2026-09-02, added after a real, confirmed
# incident: without it, Anthropic's tool_choice={"type": "any"} (propose_node's
# mode) does NOT guarantee the model only selects from the tools actually
# offered -- confirmed live, `AGENT_REGRESSED=true` correctly excluded
# lookup_order from PROPOSE_TOOLS, and Anthropic (claude-sonnet-4-5, no retry,
# not Groq) still returned a tool_use block named 'lookup_order' anyway.
# Anthropic's own docs confirm this is expected, not a bug: strict tool use
# ("Set strict: true on your tool definitions... guarantees... tool name is
# always valid, from provided tools") is the actual opt-in for the guarantee
# this whole harness had been assuming was automatic. See log/DECISIONS.md.
_ALL_PROPOSE_TOOLS = [
    {
        "name": "respond",
        "description": "Give a final answer for this category using only real information you already have.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ask_clarification",
        "description": "Ask the customer something essential you're missing that a tool call can't resolve.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
    },
    {
        "name": "lookup_order",
        "description": "Look up a single order's status, items, and delivery date.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "check_account_status",
        "description": "Look up an account's standing and order history.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_policy",
        "description": (
            "Search Dubba's policy docs (refund eligibility, return eligibility, "
            "shipping delay compensation, pricing/shipping, account suspension "
            "appeals, subscription cancellation) for chunks relevant to a question."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]

PROPOSE_TOOLS = (
    [t for t in _ALL_PROPOSE_TOOLS if t["name"] != "lookup_order"] if AGENT_REGRESSED else _ALL_PROPOSE_TOOLS
)

EVALUATE_TOOL = {
    "name": "evaluate",
    "description": "Judge whether the data gathered so far is sufficient to answer the customer.",
    "strict": True,
    "input_schema": {
        "type": "object",
        # reasoning BEFORE sufficient, deliberately -- tool-call fields generate in
        # schema order, so the model was committing to `sufficient` before it had
        # actually reasoned its way to a conclusion, then writing reasoning that
        # sometimes reversed itself ("...Therefore, this is NOT yet sufficient")
        # with no way to update the boolean it already emitted. This ordering makes
        # the model think first, then commit -- same principle as chain-of-thought.
        "properties": {
            "reasoning": {"type": "string"},
            "sufficient": {"type": "boolean"},
        },
        "required": ["reasoning", "sufficient"],
        "additionalProperties": False,
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


def _with_summary(system: str, state: TicketState) -> str:
    """Injects the rolling XML summary (if any older turns have been condensed into
    one) into a system prompt -- never into the `messages` list, which must stay a
    strict user/assistant alternation for the Anthropic API."""
    summary = state.get("conversation_summary_xml")
    if not summary:
        return system
    return (
        f"{system}\n\nCondensed summary of earlier parts of this conversation "
        f"(older turns already rolled up -- the raw messages you're given below are "
        f"only the most recent ones):\n{summary}"
    )


def ingest_message_node(state: TicketState) -> dict:
    """First node of every turn: appends the new user message to short-term memory
    and checks the token-count safety net (log/SESSION_DESIGN.md) -- a hard cap
    independent of the periodic every-N-turns summarization pass, for the case where
    a single turn (or a summarization lag) pushes the buffer past what's safe before
    a scheduled pass would fire."""
    buffer = state["short_term_buffer"] + [{"role": "user", "content": state["user_message"]}]
    token_count = llm_client.count_tokens(
        system=state.get("conversation_summary_xml", ""), messages=buffer
    )
    cap_breached = token_count >= TOKEN_CAP

    langfuse = get_client()
    with langfuse.start_as_current_observation(name="ingest-message", as_type="span") as span:
        span.update(
            input={"token_count": token_count, "cap": TOKEN_CAP},
            output={"cap_breached": cap_breached},
            level="WARNING" if cap_breached else "DEFAULT",
        )

    return {"short_term_buffer": buffer, "cap_breached": cap_breached}


def route_after_ingest(state: TicketState) -> str:
    return "session_escalate" if state["cap_breached"] else "classify"


def session_escalate_node(state: TicketState) -> dict:
    """Session-level HITL escalation -- bypasses category processing entirely for
    this turn. finalize_turn still runs afterward (forcing a summarization pass,
    since cap_breached is set) so the buffer actually shrinks instead of tripping
    this same escalation again next turn."""
    langfuse = get_client()
    with langfuse.start_as_current_observation(name="session-escalate-hitl", as_type="span") as span:
        span.update(
            input={"turn_count": state["turn_count"]},
            output=SESSION_LIMIT_MESSAGE,
            level="WARNING",
        )
    return {
        "categories": [],
        "category_index": 0,
        "pending_response_categories": [],
        "final_actions": [{"category": "session_limit", "message": SESSION_LIMIT_MESSAGE}],
    }


def finalize_turn_node(state: TicketState) -> dict:
    """The ONE place every path converges before END: composes the reply, appends it
    to short-term memory, and -- every SUMMARIZE_EVERY_N_TURNS exchanges, or
    immediately if this turn breached TOKEN_CAP -- rolls older raw turns into the XML
    summary, replacing them in state rather than letting short_term_buffer grow
    unbounded."""
    reply = _compose(state["final_actions"])
    buffer = state["short_term_buffer"] + [{"role": "assistant", "content": reply}]
    turn_count = state["turn_count"] + 1
    summary = state.get("conversation_summary_xml", "")

    should_summarize = state.get("cap_breached", False) or turn_count % SUMMARIZE_EVERY_N_TURNS == 0
    if should_summarize and len(buffer) > RAW_TURNS_KEPT_AFTER_SUMMARY:
        to_condense, buffer = buffer[:-RAW_TURNS_KEPT_AFTER_SUMMARY], buffer[-RAW_TURNS_KEPT_AFTER_SUMMARY:]
        langfuse = get_client()
        with langfuse.start_as_current_observation(name="rolling-summarize", as_type="span") as span:
            span.update(
                input={
                    "turn_count": turn_count,
                    "turns_condensed": len(to_condense),
                    "forced_by_cap": state.get("cap_breached", False),
                }
            )
            summary = summarize_turns(
                summary, to_condense, state["session_tool_log"], state["session_permission_denials"]
            )
            span.update(output=summary)

    return {
        "short_term_buffer": buffer,
        "conversation_summary_xml": summary,
        "turn_count": turn_count,
        "reply": reply,
        "cap_breached": False,
    }


def classify_node(state: TicketState) -> dict:
    langfuse = get_client()
    with langfuse.start_as_current_observation(name="classify-intent", as_type="generation") as gen:
        gen.update(input=state["user_message"])
        response = get_llm_client().create(
            system=_with_summary(prompts.CLASSIFY_SYSTEM_PROMPT, state),
            messages=state["short_term_buffer"],
            tools=[CLASSIFY_TOOL],
            force_tool="classify",
        )
        categories = response.tool_calls[0].input["categories"]
        gen.update(
            output=categories,
            model=response.model,
            metadata={"provider": response.provider},
            usage_details={"input": response.input_tokens, "output": response.output_tokens},
        )

    return {
        "categories": categories,
        "category_index": 0,
        "tool_call_count": 0,
        "turn_tool_results": [],
        "pending_response_categories": [],
        "final_actions": [],
    }


def propose_node(state: TicketState) -> dict:
    category = state["categories"][state["category_index"]]
    account = state["account"]
    system = _with_summary(
        f"{prompts.PROPOSE_SYSTEM_PROMPT}\n\n"
        f"Category you are handling right now: {category}\n"
        f"Authenticated customer ID: {state['customer_id']}\n"
        f"This customer's order IDs, oldest first: {account['order_ids']}\n"
        f"Account standing: {account['standing']}\n"
        f"{format_prior_tickets(state['prior_tickets'])}\n"
        f"Tool results gathered so far THIS TURN, across all categories (reuse "
        f"anything relevant here instead of re-fetching it): {state['turn_tool_results']}",
        state,
    )

    langfuse = get_client()
    with langfuse.start_as_current_observation(name="propose-action", as_type="generation") as gen:
        gen.update(input=state["user_message"])
        response = get_llm_client().create(
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
            metadata={"provider": response.provider},
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

    log_entry = {
        "tool": tool_name,
        "arguments": arguments,
        "result": result,
        "category": category,
        "turn": state["turn_count"],
    }
    return {
        "turn_tool_results": state["turn_tool_results"]
        + [{"tool": tool_name, "arguments": arguments, "result": result, "gathered_for_category": category}],
        "tool_call_count": state["tool_call_count"] + 1,
        "proposed_action": None,
        "session_tool_log": state["session_tool_log"] + [log_entry],
    }


def evaluate_node(state: TicketState) -> dict:
    category = state["categories"][state["category_index"]]
    system = (
        f"{prompts.EVALUATE_SYSTEM_PROMPT}\n\n"
        f"Category: {category}\n"
        f"Gathered data so far, THIS TURN, across all categories: {state['turn_tool_results']}"
    )

    langfuse = get_client()
    with langfuse.start_as_current_observation(name="evaluate-sufficiency", as_type="generation") as gen:
        gen.update(input=state["user_message"])
        response = get_llm_client().create(
            system=system,
            messages=[{"role": "user", "content": state["user_message"]}],
            tools=[EVALUATE_TOOL],
            force_tool="evaluate",
        )
        judgment = response.tool_calls[0].input
        gen.update(
            output=judgment,
            model=response.model,
            metadata={"provider": response.provider},
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
    """Does NOT write customer-facing prose -- that's finalize_node's job, once,
    after every category has been through this loop. ask_clarification is the one
    exception: it's short, category-specific, and propose already wrote its exact
    text, so there's nothing to defer or dedupe."""
    category = state["categories"][state["category_index"]]
    action = state["proposed_action"]
    is_clarification = action is not None and action["action_type"] == "ask_clarification"

    langfuse = get_client()
    with langfuse.start_as_current_observation(name="respond-decision", as_type="span") as span:
        span.update(
            input=action,
            output={"outcome": "ask_clarification" if is_clarification else "deferred_to_finalize"},
        )

    if is_clarification:
        return _advance(state, category, action["message"])

    return {
        "pending_response_categories": state["pending_response_categories"] + [category],
        "category_index": state["category_index"] + 1,
        "proposed_action": None,
    }


def finalize_node(state: TicketState) -> dict:
    """The ONE LLM call that writes customer-facing prose, covering every category
    in pending_response_categories together. This is what stops two categories about
    the same underlying issue (e.g. delivery_issue + refund_request for one damaged
    candle) from each independently producing near-identical text -- all gathering
    (propose/execute_tool/evaluate) for every category is done before this runs."""
    pending = state["pending_response_categories"]
    system = (
        f"{prompts.RESPOND_SYSTEM_PROMPT}\n\n"
        f"Categories to answer together, in ONE reply: {pending}\n"
        f"Gathered data, THIS TURN, across all categories: {state['turn_tool_results']}"
    )

    # Continuity check: only surfaces prior-ticket context when this turn's order
    # already came up in a past session (memory/store.py's relevant_prior_summary) --
    # not on every turn regardless of relevance, and only the condensed narrative
    # (not the full 7-field XML), to keep this prompt small.
    order_id = derive_order_id(state["session_tool_log"], {})
    prior_note = relevant_prior_summary(state["prior_tickets"], order_id)
    if prior_note:
        system += (
            f"\n\nThis order ({order_id}) was also the subject of an earlier support "
            f"ticket. Prior context, for continuity -- reference it only if actually "
            f"relevant to what the customer is asking now: {prior_note}"
        )

    system = _with_summary(system, state)

    langfuse = get_client()
    with langfuse.start_as_current_observation(name="finalize-response", as_type="generation") as gen:
        gen.update(input=state["user_message"])
        response = get_llm_client().create(system=system, messages=state["short_term_buffer"])
        message = response.text
        gen.update(
            output=message,
            model=response.model,
            metadata={"provider": response.provider},
            usage_details={"input": response.input_tokens, "output": response.output_tokens},
        )

    label = " & ".join(c.replace("_", " ").title() for c in pending)
    return {"final_actions": state["final_actions"] + [{"category": label, "message": message}]}


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

    denial_entry = {
        "tool": action["action_type"],
        "category": category,
        "reason": reason,
        "turn": state["turn_count"],
    }
    return {
        **_advance(state, category, PERMISSION_DENIED_MESSAGE),
        "session_permission_denials": state["session_permission_denials"] + [denial_entry],
    }


def route_after_category(state: TicketState) -> str:
    if state["category_index"] < len(state["categories"]):
        return "propose"
    if state["pending_response_categories"]:
        return "finalize"
    return "finalize_turn"


def build_graph(checkpointer):
    workflow = StateGraph(TicketState)
    workflow.add_node("ingest_message", ingest_message_node)
    workflow.add_node("classify", classify_node)
    workflow.add_node("propose", propose_node)
    workflow.add_node("execute_tool", execute_tool_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("respond", respond_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("escalate", escalate_node)
    workflow.add_node("reject_tool", reject_tool_node)
    workflow.add_node("session_escalate", session_escalate_node)
    workflow.add_node("finalize_turn", finalize_turn_node)

    workflow.add_edge(START, "ingest_message")
    workflow.add_conditional_edges(
        "ingest_message",
        route_after_ingest,
        {"classify": "classify", "session_escalate": "session_escalate"},
    )
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
    workflow.add_conditional_edges(
        "respond",
        route_after_category,
        {"propose": "propose", "finalize": "finalize", "finalize_turn": "finalize_turn"},
    )
    workflow.add_conditional_edges(
        "escalate",
        route_after_category,
        {"propose": "propose", "finalize": "finalize", "finalize_turn": "finalize_turn"},
    )
    workflow.add_conditional_edges(
        "reject_tool",
        route_after_category,
        {"propose": "propose", "finalize": "finalize", "finalize_turn": "finalize_turn"},
    )
    workflow.add_edge("finalize", "finalize_turn")
    workflow.add_edge("session_escalate", "finalize_turn")
    workflow.add_edge("finalize_turn", END)

    return workflow.compile(checkpointer=checkpointer)


def _compose(actions: list[dict]) -> str:
    if len(actions) == 1:
        return actions[0]["message"]
    parts = []
    for action in actions:
        label = action["category"].replace("_", " ").title()
        parts.append(f"[{label}]\n{action['message']}")
    return "\n\n".join(parts)


def _bootstrap_state(session: dict, user_message: str) -> TicketState:
    """Full seed state, used only on a thread_id's (session_id's) first turn --
    every turn after this one passes just {"user_message": ...} and lets the
    checkpointer carry session_id/customer_id/account/prior_tickets/
    short_term_buffer/conversation_summary_xml/turn_count forward untouched."""
    return {
        "session_id": session["session_id"],
        "customer_id": session["customer_id"],
        "account": session["account"],
        "prior_tickets": session["prior_tickets"],
        "short_term_buffer": [],
        "conversation_summary_xml": "",
        "turn_count": 0,
        "user_message": user_message,
        "categories": [],
        "category_index": 0,
        "tool_call_count": 0,
        "turn_tool_results": [],
        "proposed_action": None,
        "pending_response_categories": [],
        "final_actions": [],
        "sufficient": False,
        "cap_breached": False,
        "reply": "",
        "session_tool_log": [],
        "session_permission_denials": [],
    }


async def get_session_state(session_id: str) -> dict:
    """Public read of a thread's current checkpointed state, keyed the same way
    run_turn() keys writes (thread_id = session_id). Returns {} if the session
    has no checkpoint yet (never started, or the checkpointer was reset). Used by
    api.py's /close endpoint -- the HTTP surface is stateless between requests,
    so it has no other way to see what a session's short_term_buffer/summary
    actually is without asking the checkpointer directly."""
    assert _compiled_graph is not None, "call graph.init_graph(checkpointer) before get_session_state"
    existing = await _compiled_graph.aget_state({"configurable": {"thread_id": session_id}})
    return existing.values or {}


async def run_turn(session: dict, user_message: str) -> str:
    assert _compiled_graph is not None, "call graph.init_graph(checkpointer) before run_turn"
    langfuse = get_client()
    config = {"configurable": {"thread_id": session["session_id"]}}

    with langfuse.start_as_current_observation(name="ticket-turn", as_type="span") as trace:
        trace.update(input=user_message)
        with propagate_attributes(user_id=session["customer_id"], session_id=session["session_id"]):
            existing = await _compiled_graph.aget_state(config)
            turn_input: dict = (
                {"user_message": user_message} if existing.values else _bootstrap_state(session, user_message)
            )

            final_state = await _compiled_graph.ainvoke(turn_input, config)
            trace.update(metadata={"categories": ",".join(final_state["categories"])})
            reply = final_state["reply"]
        trace.update(output=reply)

    # Session-scoped fields now live in the checkpoint (keyed by thread_id); mirrored
    # back into `session` so main.py's close-time save_ticket_summary (and loop.py's
    # untouched session-dict path) keep working from one consistent shape.
    session["short_term_buffer"] = final_state["short_term_buffer"]
    session["conversation_summary_xml"] = final_state.get("conversation_summary_xml", "")
    session["turn_count"] = final_state.get("turn_count", 0)
    session["session_tool_log"] = final_state.get("session_tool_log", [])
    session["session_permission_denials"] = final_state.get("session_permission_denials", [])
    langfuse.flush()
    return reply
