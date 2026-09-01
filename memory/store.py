"""
Long-term memory: prior-ticket-history store, keyed by customer_id. Postgres now
(DATABASE_URL) -- AWS RDS is the next migration (log/DECISIONS.md roadmap), and only
DATABASE_URL's value changes when that happens, not this file. Same Postgres instance
as memory/checkpointer.py's own checkpoint tables, but a separate `tickets` table --
the checkpointer's setup() never touches it and vice versa.

Short-term memory (in-conversation buffer + rolling XML summary) lives in the graph's
own checkpointed state for graph mode (harness/graph.py's TicketState, persisted via
memory/checkpointer.py), or in the plain session dict for loop mode (harness/loop.py,
which never rolls up a summary mid-session -- everything stays raw until close). Either
way, this module owns writing exactly ONE summary row to long-term storage at session
close, and reading prior summaries back at session start -- it never writes the
verbatim raw buffer, per log/SESSION_DESIGN.md.
"""

import asyncio
import os
from datetime import datetime, timezone
from xml.etree import ElementTree

import psycopg
from psycopg.rows import dict_row

from harness.summarizer import build_summary_xml, summarize_turns


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set -- see .env.example")
    return url


async def _connect() -> psycopg.AsyncConnection:
    conn = await psycopg.AsyncConnection.connect(_database_url(), row_factory=dict_row)
    async with conn.cursor() as cur:
        await cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id SERIAL PRIMARY KEY,
                customer_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                closure_reason TEXT,
                resolution_summary TEXT NOT NULL,
                order_id TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                closed_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        await cur.execute("CREATE INDEX IF NOT EXISTS idx_tickets_customer ON tickets(customer_id)")
    await conn.commit()
    return conn


def _extract_order_id(summary_xml: str) -> str | None:
    try:
        root = ElementTree.fromstring(summary_xml)
    except ElementTree.ParseError:
        return None
    order_id = (root.findtext("order_id") or "").strip()
    return order_id or None


def _final_summary(session: dict) -> str:
    """The summary to write at close: whatever's already been rolled up
    (session["conversation_summary_xml"]), with any still-raw tail
    (session["short_term_buffer"]) folded in via one last summarization pass -- never
    the verbatim buffer itself. If there's no raw tail (closed right after a rolling
    pass fired), no forced re-summarization of already-summarized content."""
    existing = session.get("conversation_summary_xml", "") or ""
    raw_tail = session.get("short_term_buffer") or []

    if raw_tail:
        return summarize_turns(
            existing, raw_tail, session.get("session_tool_log", []), session.get("session_permission_denials", [])
        )
    if existing:
        return existing
    return build_summary_xml({})  # nothing was ever said -- empty summary, no LLM call needed


async def save_ticket_summary(session: dict, retries: int = 3) -> None:
    """Write-with-retry -- long-term history must never be silently dropped (see
    log/WHY.md). Summarization happens once, outside the retry loop (an LLM call
    failing isn't a transient DB error, retrying it here would just waste calls);
    only the DB write itself is retried, with exponential backoff."""
    summary_xml = _final_summary(session)
    order_id = _extract_order_id(summary_xml)

    delay = 0.5
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            conn = await _connect()
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        INSERT INTO tickets
                            (customer_id, session_id, closure_reason, resolution_summary, order_id, closed_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            session["customer_id"],
                            session["session_id"],
                            session.get("closure_reason"),
                            summary_xml,
                            order_id,
                            datetime.now(timezone.utc),
                        ),
                    )
                await conn.commit()
            finally:
                await conn.close()
            return
        except psycopg.OperationalError as exc:
            last_error = exc
            await asyncio.sleep(delay)
            delay *= 2

    raise RuntimeError(f"failed to write long-term memory after {retries} attempts") from last_error


async def get_customer_history(customer_id: str, limit: int = 3) -> list[dict]:
    conn = await _connect()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT session_id, closure_reason, resolution_summary, order_id, created_at
                FROM tickets
                WHERE customer_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (customer_id, limit),
            )
            rows = await cur.fetchall()
    finally:
        await conn.close()

    return [
        {
            "session_id": r["session_id"],
            "closure_reason": r["closure_reason"],
            "summary_xml": r["resolution_summary"],
            "order_id": r["order_id"],
            "created_at": r["created_at"].isoformat(),
        }
        for r in rows
    ]


def relevant_prior_summary(prior_tickets: list[dict], order_id: str | None) -> str:
    """Returns just the <convo_summary> text from whichever prior ticket concerns the
    SAME order as the current turn (order_id, from harness/summarizer.py's
    derive_order_id -- i.e. this turn's most recent lookup_order call) -- empty string
    if there's no order_id yet, or no prior ticket matches it. This is the
    "continuity" check: only surfaces prior history when the current query is
    actually about an order that came up before, not on every turn regardless of
    relevance.

    Extracts one field, not the full 7-field XML -- keeps finalize_node's prompt
    small (it's writing customer-facing prose, not doing tool-call reasoning, so it
    doesn't need tool_calls_made/tool_results/permission_checks from a past ticket).
    Uses ElementTree, not a raw string/regex scan -- build_summary_xml() escapes
    content (xml.sax.saxutils.escape), so a naive regex would leave HTML entities
    (e.g. "&amp;") in the extracted text instead of decoding them back."""
    if not order_id:
        return ""
    for t in prior_tickets:
        if t.get("order_id") != order_id:
            continue
        try:
            root = ElementTree.fromstring(t["summary_xml"])
        except ElementTree.ParseError:
            continue
        summary = root.findtext("convo_summary")
        if summary:
            return summary
    return ""


def format_prior_tickets(prior_tickets: list[dict]) -> str:
    """Compact, prompt-ready rendition -- used by both harness implementations."""
    if not prior_tickets:
        return "No prior support tickets on file for this customer."
    lines = [f"{len(prior_tickets)} prior support ticket(s) on file, most recent first:"]
    for t in prior_tickets:
        lines.append(f"- {t['created_at']} (closed: {t['closure_reason']}): {t['summary_xml']}")
    return "\n".join(lines)
