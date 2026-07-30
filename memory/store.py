"""
Long-term memory: prior-ticket-history store, keyed by customer_id. SQLite for this
phase -- Postgres, then AWS RDS, are later-week migrations (log/DECISIONS.md). Only
this file changes when that swap happens.

Short-term memory (in-conversation buffer) already lives in session["short_term_buffer"],
managed by harness/loop.py and harness/graph.py. This module owns writing a summary
of it to long-term storage at session close, and reading prior summaries back at
session start.

Summary is built deterministically from session state (turns, closure_reason), not
via an LLM call -- the harness already has structured turn data, so there's nothing
"lightweight NLP or a cheap LLM call" would add over just formatting what's there.
"""

import sqlite3
import time
from pathlib import Path
from xml.sax.saxutils import escape

DB_PATH = Path(__file__).resolve().parent / "long_term_memory.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ticket_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            closure_reason TEXT,
            summary_xml TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ticket_history_customer ON ticket_history(customer_id)"
    )
    return conn


def build_summary_xml(session: dict) -> str:
    turns = "".join(
        f'<turn role="{escape(t["role"])}">{escape(t["content"])}</turn>'
        for t in session["short_term_buffer"]
    )
    return (
        f'<ticket_summary session_id="{escape(session["session_id"])}" '
        f'closure_reason="{escape(session.get("closure_reason") or "unknown")}">'
        f"<transcript>{turns}</transcript>"
        f"</ticket_summary>"
    )


def save_ticket_summary(session: dict, retries: int = 3) -> None:
    """Write-with-retry -- long-term history must never be silently dropped (see log/WHY.md)."""
    summary_xml = build_summary_xml(session)
    delay = 0.5
    last_error: Exception | None = None

    for _ in range(retries):
        try:
            conn = _connect()
            try:
                conn.execute(
                    "INSERT INTO ticket_history (customer_id, session_id, closure_reason, summary_xml) "
                    "VALUES (?, ?, ?, ?)",
                    (session["customer_id"], session["session_id"], session.get("closure_reason"), summary_xml),
                )
                conn.commit()
            finally:
                conn.close()
            return
        except sqlite3.OperationalError as exc:
            last_error = exc
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(f"failed to write long-term memory after {retries} attempts") from last_error


def get_ticket_history(customer_id: str, limit: int = 3) -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT session_id, closure_reason, summary_xml, created_at FROM ticket_history "
            "WHERE customer_id = ? ORDER BY created_at DESC LIMIT ?",
            (customer_id, limit),
        ).fetchall()
    finally:
        conn.close()

    return [
        {"session_id": r[0], "closure_reason": r[1], "summary_xml": r[2], "created_at": r[3]}
        for r in rows
    ]


def format_prior_tickets(prior_tickets: list[dict]) -> str:
    """Compact, prompt-ready rendition -- used by both harness implementations."""
    if not prior_tickets:
        return "No prior support tickets on file for this customer."
    lines = [f"{len(prior_tickets)} prior support ticket(s) on file, most recent first:"]
    for t in prior_tickets:
        lines.append(f"- {t['created_at']} (closed: {t['closure_reason']}): {t['summary_xml']}")
    return "\n".join(lines)
