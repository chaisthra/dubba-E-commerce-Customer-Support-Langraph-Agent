"""
The while-loop harness. LLM proposes an action; this file decides whether it
executes -- schema.validate_action() then permissions.check_action_permission() run
on every proposed action before anything happens. No path skips either check.

Per user turn: classify (all categories, upfront) -> for each category, sequentially:
decide -> validate -> permission-check -> execute -> compose all outputs into one
reply. Every step is traced in Langfuse: one trace per turn, one span per category.
"""

import uuid

from langfuse import get_client, propagate_attributes

from harness import llm_client, permissions, prompts, schema
from memory.store import format_prior_tickets, get_ticket_history

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

DECIDE_TOOL = {
    "name": "decide",
    "description": "Propose exactly one action for this ticket category.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action_type": {"type": "string", "enum": sorted(schema.VALID_ACTION_TYPES)},
            "category": {"type": "string", "enum": sorted(schema.VALID_CATEGORIES)},
            "message": {"type": "string"},
        },
        "required": ["action_type", "category", "message"],
    },
}

REJECTED_ACTION_MESSAGE = (
    "Sorry, I hit an internal issue handling that part of your request. "
    "Could you rephrase it?"
)


def new_session(customer_id: str, account: dict) -> dict:
    return {
        "session_id": str(uuid.uuid4()),
        "customer_id": customer_id,
        "account": account,
        "short_term_buffer": [],
        "closure_reason": None,
        "prior_tickets": get_ticket_history(customer_id),
    }


def _classify(session: dict, langfuse) -> list[str]:
    with langfuse.start_as_current_observation(
        name="classify-intent", as_type="generation", model=llm_client.PRIMARY_MODEL
    ) as gen:
        gen.update(input=session["short_term_buffer"][-1]["content"])
        response = llm_client.complete(
            system=prompts.CLASSIFY_SYSTEM_PROMPT,
            messages=session["short_term_buffer"],
            tools=[CLASSIFY_TOOL],
            force_tool="classify",
        )
        categories = response.tool_calls[0].input["categories"]
        gen.update(
            output=categories,
            model=response.model,
            usage_details={
                "input": response.input_tokens,
                "output": response.output_tokens,
            },
        )
        return categories


def _decide(session: dict, category: str, langfuse) -> dict:
    account = session["account"]
    system = (
        f"{prompts.DECIDE_SYSTEM_PROMPT}\n\n"
        f"The category you are handling right now: {category}\n"
        f"Authenticated customer ID: {session['customer_id']}\n"
        f"Known order IDs for this customer: {account['order_ids']}\n"
        f"Account standing: {account['standing']}\n"
        f"{format_prior_tickets(session['prior_tickets'])}"
    )
    with langfuse.start_as_current_observation(
        name="decide-action", as_type="generation", model=llm_client.PRIMARY_MODEL
    ) as gen:
        gen.update(input=session["short_term_buffer"][-1]["content"])
        response = llm_client.complete(
            system=system,
            messages=session["short_term_buffer"],
            tools=[DECIDE_TOOL],
            force_tool="decide",
        )
        action = response.tool_calls[0].input
        gen.update(
            output=action,
            model=response.model,
            usage_details={
                "input": response.input_tokens,
                "output": response.output_tokens,
            },
        )
        return action


def _check_action(action: dict, session: dict, langfuse) -> tuple[bool, str]:
    with langfuse.start_as_current_observation(
        name="check-action", as_type="span"
    ) as check_span:
        check_span.update(input=action)

        shape_ok, shape_reason = schema.validate_action(action)
        if not shape_ok:
            check_span.update(output={"allowed": False, "reason": shape_reason}, level="ERROR")
            return False, shape_reason

        allowed, reason = permissions.check_action_permission(action, session)
        check_span.update(output={"allowed": allowed, "reason": reason})
        return allowed, reason


def _compose(actions: list[dict]) -> str:
    if len(actions) == 1:
        return actions[0]["message"]
    parts = []
    for action in actions:
        label = action["category"].replace("_", " ").title()
        parts.append(f"[{label}]\n{action['message']}")
    return "\n\n".join(parts)


def handle_turn(session: dict, user_message: str) -> str:
    langfuse = get_client()
    session["short_term_buffer"].append({"role": "user", "content": user_message})

    with langfuse.start_as_current_observation(name="ticket-turn", as_type="span") as trace:
        trace.update(input=user_message)
        with propagate_attributes(
            user_id=session["customer_id"],
            session_id=session["session_id"],
        ):
            categories = _classify(session, langfuse)
            trace.update(metadata={"categories": ",".join(categories)})

            actions = []
            for category in categories:
                with langfuse.start_as_current_observation(
                    name=f"category:{category}", as_type="span"
                ):
                    action = _decide(session, category, langfuse)
                    allowed, reason = _check_action(action, session, langfuse)
                    if allowed:
                        actions.append(action)
                    else:
                        actions.append(
                            {
                                "action_type": "respond",
                                "category": category,
                                "message": REJECTED_ACTION_MESSAGE,
                            }
                        )

            reply = _compose(actions)
        trace.update(output=reply)

    session["short_term_buffer"].append({"role": "assistant", "content": reply})
    langfuse.flush()
    return reply
