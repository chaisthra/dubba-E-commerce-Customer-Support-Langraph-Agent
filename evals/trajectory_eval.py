"""
Rule-based trajectory eval (Assignment 2, section 2.2).

Runs each ticket in evals/golden_dataset.json through the real agent
(harness.graph.run_turn) and checks whether the required tools for that ticket
type actually got called -- a SUPERSET check (extra calls are fine, missing ones
fail), matching the assignment's own build guide.

Deliberately NOT a final-answer eval -- this never looks at the reply text, only at
session_tool_log (harness/graph.py's session-scoped, append-only tool-call record,
added this same week). A wrong-but-plausible-sounding final answer with a broken
trajectory is exactly the "confident wrong path" failure mode this eval exists to
catch -- see log/learnings/2026-08-28-confident-wrong-path-return-request.md for a
real case found this way.

prior_tickets is forced empty for every ticket, regardless of what's actually in
Postgres for that customer by the time this runs. Real prior-ticket history would
make this eval non-deterministic -- a ticket could skip a required lookup because
prior-ticket continuity (harness/graph.py's finalize_node) already "knows" the
answer from an unrelated earlier eval run or real conversation. A reproducible
regression check can't depend on what order eval runs happened to occur in.

Needs the full stack running: Postgres (docker compose up -d), the MCP subprocess
(spawned automatically), and real Anthropic/Groq API access via .env -- same
requirements as `python main.py`.

Usage:
    python evals/trajectory_eval.py
    python evals/trajectory_eval.py --dataset evals/golden_dataset.json --out evals/trajectory_eval_results.json
"""

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from harness import graph  # noqa: E402
from harness.mcp_client import MCPClient  # noqa: E402
from mcp_server.mock_data import find_account_by_email  # noqa: E402
from memory.checkpointer import PostgresCheckpointer  # noqa: E402


def _empty_result(ticket: dict, error: str | None = None) -> dict:
    return {
        "id": ticket["id"],
        "query": ticket["query"],
        "expected_trajectory": ticket["expected_trajectory"],
        "actual_trajectory": [],
        "missing": list(ticket["expected_trajectory"]),
        "extra": [],
        "passed": False,
        "reply": None,
        "error": error,
    }


async def run_ticket(ticket: dict) -> dict:
    account = find_account_by_email(ticket["customer_email"])
    if account is None:
        return _empty_result(ticket, error=f"no account found for {ticket['customer_email']!r}")

    session = {
        "session_id": str(uuid.uuid4()),
        "customer_id": account["customer_id"],
        "account": account,
        "short_term_buffer": [],
        "conversation_summary_xml": "",
        "closure_reason": None,
        "prior_tickets": [],  # forced empty -- see module docstring
    }

    try:
        reply = await graph.run_turn(session, ticket["query"])
    except Exception as exc:
        return _empty_result(ticket, error=f"{type(exc).__name__}: {exc}")

    actual_tools = [entry["tool"] for entry in session.get("session_tool_log", [])]
    expected = set(ticket["expected_trajectory"])
    actual_set = set(actual_tools)
    missing = sorted(expected - actual_set)
    extra = sorted(actual_set - expected)

    return {
        "id": ticket["id"],
        "query": ticket["query"],
        "expected_trajectory": ticket["expected_trajectory"],
        "actual_trajectory": actual_tools,
        "missing": missing,
        "extra": extra,
        "passed": not missing,
        "reply": reply,
        "error": None,
    }


async def run_all(dataset_path: Path) -> list[dict]:
    tickets = json.loads(dataset_path.read_text())

    mcp_client = MCPClient()
    await mcp_client.start()
    graph.set_mcp_client(mcp_client)

    checkpointer = PostgresCheckpointer()
    await checkpointer.start()
    graph.init_graph(checkpointer.saver)

    results = []
    try:
        for ticket in tickets:
            print(f"running {ticket['id']}...", flush=True)
            result = await run_ticket(ticket)
            results.append(result)
            status = "PASS" if result["passed"] else "FAIL"
            line = f"  [{status}] expected={result['expected_trajectory']} actual={result['actual_trajectory']}"
            if result["missing"]:
                line += f" MISSING={result['missing']}"
            if result["error"]:
                line += f" ERROR={result['error']}"
            print(line)
    finally:
        await mcp_client.close()
        await checkpointer.close()

    return results


def report(results: list[dict]) -> float:
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    score = passed / total if total else 0.0

    print(f"\n=== {passed}/{total} passed ({score:.1%}) ===")
    failures = [r for r in results if not r["passed"]]
    if failures:
        print("Failures:")
        for r in failures:
            detail = f"missing {r['missing']}"
            if r["error"]:
                detail += f" (error: {r['error']})"
            print(f"  - {r['id']}: {detail}")

    return score


async def main(dataset_path: Path, out_path: Path) -> int:
    results = await run_all(dataset_path)
    score = report(results)

    out_path.write_text(
        json.dumps(
            {"score": score, "passed": sum(1 for r in results if r["passed"]), "total": len(results), "results": results},
            indent=2,
        )
    )
    print(f"\nWrote results to {out_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rule-based trajectory eval (superset of required tools per ticket)")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "evals" / "golden_dataset.json")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "evals" / "trajectory_eval_results.json")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.dataset, args.out)))
