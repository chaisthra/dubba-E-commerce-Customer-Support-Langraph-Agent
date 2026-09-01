"""
LLM-as-judge (Assignment 2, section 2.3).

Fact-checking rubric, not a vague "rate this 1-5" -- for every specific, checkable
claim in the agent's actual reply (a number, a date, a policy rule, an eligibility
determination, a dollar amount, a process step), does it hold up against REAL
reference material? Reference = the actual policy doc text (loaded from
rag/policy_docs/ via each ticket's expected_source), the real order record (loaded
fresh from mcp_server.mock_data.get_order() via each ticket's order_id, when
present), and the golden dataset's independently-authored expected_answer -- never
the agent's own reply used as its own reference, per the assignment's build guide.

The order record is included on purpose, not just the hand-written expected_answer:
run 1 of this script judged several order-status replies as unsupported for
including real scent/variant details (e.g. "Mona Lisa's Smirk (Vanilla Musk)")
that were 100% accurate -- the agent had them from the real lookup_order result,
but my own expected_answer text had abbreviated them away, so the judge had no way
to know they were true. Giving the judge the actual order record fixes that at the
reference level rather than reference the agent's own reply, which would defeat the
point of an independent check. See log/learnings/2026-08-29-llm-judge-run1-findings.md.

Uses GroqProvider directly (same as harness/summarizer.py), which already bakes in
temperature=0 (harness/llm_provider.py). Plain-text output, not forced tool-calling
-- this session already found forced tool_choice unreliable on Groq (see
log/DECISIONS.md's summarizer rewrite; a model disobeying a forced tool name
crashed a session close), no reason to reintroduce that fragility for a judge.

Session 3's own finding, flagged as a common pitfall: judge scores are
non-deterministic even at temperature=0 (batched inference isn't perfectly
reproducible). Runs each ticket through the judge 3 times and reports every run,
not just one -- never treat a single run as a verdict.

Input: evals/trajectory_eval_results.json (the actual replies from a real
trajectory-eval run) + evals/golden_dataset.json (expected_answer/expected_source
as reference). Run evals/trajectory_eval.py first if results don't exist yet.

Usage:
    python evals/llm_judge.py
    python evals/llm_judge.py --runs 3 --results evals/trajectory_eval_results.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from harness.llm_provider import GroqProvider  # noqa: E402
from mcp_server.mock_data import get_order  # noqa: E402

JUDGE_SYSTEM_PROMPT = """You are a strict fact-checking judge for an e-commerce \
support agent's responses. You are NOT rating how good the response sounds -- you \
are checking whether every specific, checkable claim in it (a number, a date, a \
policy rule, an eligibility determination, a dollar amount, a process step) is \
actually supported by the REFERENCE MATERIAL you're given. A confident, \
well-written, plausible-sounding response with an unsupported or contradicted \
claim must score LOW -- fluency and tone are not what you're scoring.

Score 1-5:
5 = every specific claim in the response is directly and clearly supported by the \
reference material. No fabricated or unsupported details.
4 = all claims are supported; only a minor phrasing/emphasis difference, not a \
factual one.
3 = mostly supported, but one relatively minor claim is unverifiable or imprecise \
against the reference.
2 = at least one significant claim is unsupported by or contradicts the reference \
(a wrong number, a wrong eligibility determination, an invented detail, a wrong \
order being discussed).
1 = the response's core answer to the customer's actual question is unsupported \
by, or contradicts, the reference material -- or the response addresses something \
materially different from what the reference/question is actually about.

Reply in EXACTLY this format, nothing else, nothing before or after it:
REASONING: <one or two sentences, naming the SPECIFIC claim(s) you checked and \
whether the reference supports them>
SCORE: <a single integer, 1-5>"""


def _build_reference(ticket: dict) -> str:
    parts = [f"Independently-authored reference answer (ground truth, not the agent's words):\n{ticket['expected_answer']}"]

    order_id = ticket.get("order_id")
    if order_id:
        order = get_order(order_id)
        if order is not None:
            parts.append(
                f"Real order record for {order_id} (fetched fresh, authoritative -- item names, "
                f"scents, dates, and computed eligibility fields here are all true even if the "
                f"reference answer above didn't spell every one of them out):\n{json.dumps(order, indent=2)}"
            )

    source = ticket.get("expected_source")
    if source:
        doc_path = PROJECT_ROOT / "rag" / "policy_docs" / source
        if doc_path.exists():
            parts.append(f"Full source policy document ({source}):\n{doc_path.read_text()}")

    return "\n\n".join(parts)


def judge_once(judge: GroqProvider, query: str, reference: str, response: str) -> tuple[int | None, str]:
    user_content = (
        f"CUSTOMER QUESTION:\n{query}\n\n"
        f"REFERENCE MATERIAL:\n{reference}\n\n"
        f"AGENT'S ACTUAL RESPONSE TO CHECK:\n{response}"
    )
    result = judge.create(system=JUDGE_SYSTEM_PROMPT, messages=[{"role": "user", "content": user_content}])
    text = (result.text or "").strip()
    match = re.search(r"SCORE:\s*([1-5])", text)
    score = int(match.group(1)) if match else None
    return score, text


def main(results_path: Path, dataset_path: Path, out_path: Path, runs: int) -> int:
    eval_results = {r["id"]: r for r in json.loads(results_path.read_text())["results"]}
    tickets = {t["id"]: t for t in json.loads(dataset_path.read_text())}

    judge = GroqProvider()
    ticket_reports = []

    for ticket_id, ticket in tickets.items():
        eval_result = eval_results.get(ticket_id)
        if eval_result is None or eval_result.get("reply") is None:
            print(f"[SKIP] {ticket_id}: no reply available (trajectory run errored or missing)")
            continue

        reference = _build_reference(ticket)
        reply = eval_result["reply"]

        print(f"judging {ticket_id} ({runs} runs)...", flush=True)
        run_scores = []
        run_details = []
        for i in range(runs):
            score, raw = judge_once(judge, ticket["query"], reference, reply)
            run_scores.append(score)
            run_details.append(raw)
            print(f"  run {i + 1}: score={score}")

        valid_scores = [s for s in run_scores if s is not None]
        avg = round(mean(valid_scores), 2) if valid_scores else None
        spread = (max(valid_scores) - min(valid_scores)) if len(valid_scores) > 1 else 0

        ticket_reports.append(
            {
                "id": ticket_id,
                "query": ticket["query"],
                "reply": reply,
                "scores": run_scores,
                "average": avg,
                "spread": spread,
                "run_details": run_details,
            }
        )
        print(f"  -> average={avg} spread={spread}\n")

    print("=== LLM-as-judge summary ===")
    for r in ticket_reports:
        flag = " (HIGH VARIANCE)" if r["spread"] and r["spread"] >= 2 else ""
        print(f"{r['id']}: scores={r['scores']} avg={r['average']}{flag}")

    dataset_avg = round(mean(r["average"] for r in ticket_reports if r["average"] is not None), 2) if ticket_reports else None
    print(f"\nOverall average across {len(ticket_reports)} tickets, {runs} runs each: {dataset_avg}")

    out_path.write_text(json.dumps({"runs_per_ticket": runs, "overall_average": dataset_avg, "tickets": ticket_reports}, indent=2))
    print(f"\nWrote results to {out_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-judge, fact-checking rubric, multiple runs per ticket")
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "evals" / "trajectory_eval_results.json")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "evals" / "golden_dataset.json")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "evals" / "llm_judge_results.json")
    parser.add_argument("--runs", type=int, default=3, help="judge runs per ticket (non-determinism check)")
    args = parser.parse_args()
    sys.exit(main(args.results, args.dataset, args.out, args.runs))
