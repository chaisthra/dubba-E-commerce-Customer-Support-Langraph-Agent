"""
Rolling short-term-memory summarization (log/SESSION_DESIGN.md's structured XML
format). The harness builds six of the seven fields deterministically from its own
tracked state (session_tool_log, session_permission_denials, the previously-parsed
summary) -- only `convo_summary` is genuine LLM work, since compressing a narrative
is the one thing here that actually requires judgment rather than lookup/
concatenation.

This replaces an earlier version that forced ALL seven fields through a single tool
call (force_tool="summarize"). That was fragile in two ways: (1) a model disobeying
the forced tool name crashed the whole session-close write (Groq's openai/gpt-oss-20b
did exactly this in production -- see log/DECISIONS.md), and (2) five of those seven
fields were never real structured state to begin with -- short_term_buffer only ever
holds customer-facing prose, never tool calls or results, so the model was inventing
"structured" data by inferring it from text it never actually had.

Called from two places: harness/graph.py's finalize_turn node (periodic, every ~5
exchanges, or forced early on a token-cap breach) and memory/store.py's
save_ticket_summary (once, at session close, to fold any still-raw tail before
writing long-term history) -- so the same logic backs both the live rolling cap and
the final long-term write.
"""

from xml.etree import ElementTree
from xml.sax.saxutils import escape

from langfuse import get_client

from harness import prompts
from harness.llm_provider import GroqProvider

_groq = GroqProvider()

SUMMARY_FIELDS = (
    "order_id",
    "exact_complaint",
    "context",
    "tool_calls_made",
    "tool_results",
    "permission_checks",
    "convo_summary",
)


def _render_turns(turns: list[dict]) -> str:
    return "\n".join(f"{t['role']}: {t['content']}" for t in turns)


def build_summary_xml(fields: dict) -> str:
    inner = "".join(f"<{name}>{escape(fields.get(name, '') or '')}</{name}>" for name in SUMMARY_FIELDS)
    return f"<session_summary>{inner}</session_summary>"


def _parse_existing(summary_xml: str) -> dict:
    """Reads a previously-built summary back into a fields dict, so this pass can
    carry forward whatever isn't being freshly recomputed (exact_complaint, context,
    and convo_summary as the base to extend). Malformed/empty input yields an
    all-empty dict -- best-effort carry-forward, never raises."""
    if not summary_xml:
        return {}
    try:
        root = ElementTree.fromstring(summary_xml)
    except ElementTree.ParseError:
        return {}
    return {name: (root.findtext(name) or "") for name in SUMMARY_FIELDS}


def derive_order_id(session_tool_log: list[dict], prior: dict) -> str:
    """Most recent lookup_order call's order_id wins. session_tool_log is
    session-scoped (appended to, never reset per turn -- unlike turn_tool_results),
    so this is recomputed fresh from the FULL session's tool history every pass, not
    merged incrementally. Falls back to whatever a previous pass already had, for the
    rare case an order was already established before any lookup_order call.

    Public (not underscore-prefixed): also called from harness/graph.py's
    finalize_node, to check this turn's order against prior-ticket history
    (memory/store.py's relevant_prior_summary) -- same "what order is this session
    actually about" question, reused rather than re-derived."""
    order_ids = [
        entry["arguments"].get("order_id")
        for entry in session_tool_log
        if entry["tool"] == "lookup_order" and entry["arguments"].get("order_id")
    ]
    return order_ids[-1] if order_ids else prior.get("order_id", "")


def _render_tool_calls(session_tool_log: list[dict]) -> str:
    if not session_tool_log:
        return ""
    return "; ".join(f"{e['tool']}({e['arguments']}) [{e['category']}, turn {e['turn']}]" for e in session_tool_log)


def _render_tool_results(session_tool_log: list[dict]) -> str:
    if not session_tool_log:
        return ""
    return "; ".join(f"{e['tool']} -> {e['result']}" for e in session_tool_log)


def _render_permission_checks(session_tool_log: list[dict], denials: list[dict]) -> str:
    parts = []
    if session_tool_log:
        parts.append(f"{len(session_tool_log)} tool call(s) executed this session, all passed ownership checks.")
    if denials:
        parts.append(
            "; ".join(f"DENIED: {d['tool']} ({d['reason']}) [{d['category']}, turn {d['turn']}]" for d in denials)
        )
    return " ".join(parts)


def _derive_exact_complaint(prior: dict, turns_to_condense: list[dict]) -> str:
    """Deliberately verbatim, not model-paraphrased -- the field's whole point is
    "the customer's issue, in their own terms", so quoting their raw messages is more
    faithful than a summary of a summary. Carries forward whatever an earlier pass
    already captured, since raw turns are dropped from short_term_buffer once
    condensed -- this XML field is the only place that text survives afterward."""
    new_lines = [t["content"] for t in turns_to_condense if t["role"] == "user"]
    new_text = " | ".join(new_lines)
    existing = prior.get("exact_complaint", "")
    if existing and new_text:
        return f"{existing} | {new_text}"
    return existing or new_text


def _fallback_convo_summary(prior: dict, turns_to_condense: list[dict], session_tool_log: list[dict]) -> str:
    """Used only if the Groq call fails outright (every model in GroqProvider's chain
    failed). A session close must never fail on summarization just because an LLM
    call did -- deterministic, built from state alone: carry forward whatever
    narrative already existed and append a factual note instead of inventing prose."""
    note = (
        f"[condensed without model summarization -- Groq unavailable] "
        f"{len(turns_to_condense)} exchange(s) folded in; "
        f"{len(session_tool_log)} tool call(s) made this session so far."
    )
    existing = prior.get("convo_summary", "")
    return f"{existing} {note}".strip()


def _model_convo_summary(prior_convo_summary: str, turns_to_condense: list[dict]):
    """The one genuinely LLM-dependent field: real compression of prose, which a
    template can't do. Plain text completion -- no tools, no forced tool_choice, so
    there's nothing for a model to disobey, only text to read back. Returns the raw
    LLMResponse (not just text) so the caller can still log model/provider/usage to
    Langfuse. temperature=0 is baked into GroqProvider.create() itself
    (harness/llm_provider.py), not passed here."""
    user_content = (
        (f"EXISTING SUMMARY:\n{prior_convo_summary}\n\n" if prior_convo_summary else "")
        + f"NEW RAW TURNS TO FOLD IN:\n{_render_turns(turns_to_condense)}"
    )
    return _groq.create(
        system=prompts.SUMMARIZE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )


def summarize_turns(
    existing_summary_xml: str,
    turns_to_condense: list[dict],
    session_tool_log: list[dict],
    permission_denials: list[dict],
) -> str:
    """Returns an updated summary XML string folding `turns_to_condense` into
    `existing_summary_xml` (empty string if this is the first pass). Never called
    with an empty `turns_to_condense` -- callers only invoke this when there's
    actually new raw content to fold in.

    order_id / tool_calls_made / tool_results / permission_checks are computed
    deterministically from session_tool_log/permission_denials (both session-scoped,
    appended-to-never-reset -- see harness/graph.py's TicketState). exact_complaint
    carries forward + appends verbatim customer text. Only convo_summary is an actual
    LLM call, with a deterministic fallback if it fails outright."""
    prior = _parse_existing(existing_summary_xml)

    fields = {
        "order_id": derive_order_id(session_tool_log, prior),
        "exact_complaint": _derive_exact_complaint(prior, turns_to_condense),
        "context": prior.get("context", ""),
        "tool_calls_made": _render_tool_calls(session_tool_log),
        "tool_results": _render_tool_results(session_tool_log),
        "permission_checks": _render_permission_checks(session_tool_log, permission_denials),
    }

    langfuse = get_client()
    with langfuse.start_as_current_observation(name="summarize-session", as_type="generation") as gen:
        gen.update(input=_render_turns(turns_to_condense))
        try:
            response = _model_convo_summary(prior.get("convo_summary", ""), turns_to_condense)
            fields["convo_summary"] = (response.text or "").strip()
            gen.update(
                model=response.model,
                metadata={"provider": response.provider, "fallback": False},
                usage_details={"input": response.input_tokens, "output": response.output_tokens},
            )
        except Exception as exc:
            fields["convo_summary"] = _fallback_convo_summary(prior, turns_to_condense, session_tool_log)
            gen.update(level="WARNING", metadata={"fallback": True, "error": str(exc)})

        summary_xml = build_summary_xml(fields)
        gen.update(output=summary_xml)

    return summary_xml
