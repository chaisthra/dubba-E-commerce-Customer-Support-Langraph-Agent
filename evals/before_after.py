"""
Before/after report (Assignment 2, section 2.5).

Compares the CURRENT combined score (a fresh run, same as evals/combined_score.py)
against the saved baseline snapshot (evals/baseline.json +
evals/baseline_trajectory_results.json + evals/baseline_judge_results.json,
written by `python evals/combined_score.py --save-baseline`) -- prints the
aggregate before/after numbers AND, per ticket, whether its trajectory pass/fail
status or judge average changed, so "which specific tickets flipped" is an actual
answer, not just an aggregate delta.

Usage:
    python evals/before_after.py                 # fresh run vs. saved baseline
    python evals/before_after.py --after-only     # skip the fresh run, diff two
                                                    already-saved result sets
                                                    (--after-trajectory/--after-judge)
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import evals.combined_score as combined_score  # noqa: E402


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _trajectory_by_id(data: dict) -> dict:
    return {r["id"]: r for r in data.get("results", [])}


def _judge_by_id(data: dict) -> dict:
    return {t["id"]: t for t in data.get("tickets", [])}


def diff(before_trajectory: dict, before_judge: dict, after_trajectory: dict, after_judge: dict) -> list[dict]:
    before_traj_by_id = _trajectory_by_id(before_trajectory)
    after_traj_by_id = _trajectory_by_id(after_trajectory)
    before_judge_by_id = _judge_by_id(before_judge)
    after_judge_by_id = _judge_by_id(after_judge)

    all_ids = sorted(set(before_traj_by_id) | set(after_traj_by_id))
    rows = []
    for ticket_id in all_ids:
        b_traj = before_traj_by_id.get(ticket_id)
        a_traj = after_traj_by_id.get(ticket_id)
        b_judge = before_judge_by_id.get(ticket_id)
        a_judge = after_judge_by_id.get(ticket_id)

        b_passed = b_traj["passed"] if b_traj else None
        a_passed = a_traj["passed"] if a_traj else None
        b_avg = b_judge["average"] if b_judge else None
        a_avg = a_judge["average"] if a_judge else None

        trajectory_flipped = b_passed is not None and a_passed is not None and b_passed != a_passed
        judge_delta = (a_avg - b_avg) if (a_avg is not None and b_avg is not None) else None
        judge_flipped = judge_delta is not None and abs(judge_delta) >= 1.0  # a full point of judge movement, not run-to-run noise

        rows.append(
            {
                "id": ticket_id,
                "trajectory_before": b_passed,
                "trajectory_after": a_passed,
                "trajectory_flipped": trajectory_flipped,
                "judge_before": b_avg,
                "judge_after": a_avg,
                "judge_delta": round(judge_delta, 2) if judge_delta is not None else None,
                "judge_flipped": judge_flipped,
                "changed": trajectory_flipped or judge_flipped,
            }
        )
    return rows


def report(before_score: dict, after_score: dict, rows: list[dict]) -> None:
    print("=== Before/After Report ===\n")
    if before_score:
        print(
            f"BEFORE: combined={before_score.get('combined_score')} "
            f"(trajectory={before_score.get('trajectory_score_pct')}% "
            f"[{before_score.get('trajectory_passed')}/{before_score.get('trajectory_total')}], "
            f"judge={before_score.get('judge_score_pct')}% [avg {before_score.get('judge_average')}/5])"
        )
    if after_score:
        print(
            f"AFTER:  combined={after_score.get('combined_score')} "
            f"(trajectory={after_score.get('trajectory_score_pct')}% "
            f"[{after_score.get('trajectory_passed')}/{after_score.get('trajectory_total')}], "
            f"judge={after_score.get('judge_score_pct')}% [avg {after_score.get('judge_average')}/5])"
        )
    if before_score and after_score:
        drop = round(before_score.get("combined_score", 0) - after_score.get("combined_score", 0), 2)
        direction = "dropped" if drop > 0 else "improved by" if drop < 0 else "unchanged,"
        print(f"\n{'DROP' if drop > 0 else 'CHANGE'}: combined score {direction} {abs(drop)} points\n")

    changed = [r for r in rows if r["changed"]]
    print(f"=== Per-ticket: {len(changed)}/{len(rows)} tickets changed ===")
    for r in rows:
        if not r["changed"]:
            continue
        parts = [f"  {r['id']}:"]
        if r["trajectory_flipped"]:
            parts.append(f"trajectory {r['trajectory_before']} -> {r['trajectory_after']}")
        if r["judge_flipped"]:
            parts.append(f"judge {r['judge_before']} -> {r['judge_after']} ({r['judge_delta']:+})")
        print(" ".join(parts))

    if not changed:
        print("  (none)")


async def main(args: argparse.Namespace) -> int:
    baseline_score = _load(PROJECT_ROOT / "evals" / "baseline.json")
    baseline_trajectory = _load(PROJECT_ROOT / "evals" / "baseline_trajectory_results.json")
    baseline_judge = _load(PROJECT_ROOT / "evals" / "baseline_judge_results.json")

    if not baseline_score:
        print("No baseline found -- run `python evals/combined_score.py --save-baseline` first.")
        return 1

    if args.after_only:
        after_trajectory = _load(Path(args.after_trajectory))
        after_judge = _load(Path(args.after_judge))
        after_score = None  # not recomputed -- caller is diffing two pre-existing result sets directly
    else:
        after_score = await combined_score.compute_score(
            args.dataset, args.trajectory_out, args.judge_out, args.judge_runs
        )
        after_trajectory = _load(args.trajectory_out)
        after_judge = _load(args.judge_out)

    rows = diff(baseline_trajectory, baseline_judge, after_trajectory, after_judge)
    report(baseline_score, after_score, rows)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Before/after eval comparison against the saved baseline")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "evals" / "golden_dataset.json")
    parser.add_argument("--trajectory-out", type=Path, default=PROJECT_ROOT / "evals" / "trajectory_eval_results.json")
    parser.add_argument("--judge-out", type=Path, default=PROJECT_ROOT / "evals" / "llm_judge_results.json")
    parser.add_argument("--judge-runs", type=int, default=combined_score.JUDGE_RUNS_FOR_GATE)
    parser.add_argument("--after-only", action="store_true", help="skip running a fresh eval; diff --after-trajectory/--after-judge against the saved baseline")
    parser.add_argument("--after-trajectory", type=str, default=None, help="with --after-only: path to a trajectory_eval_results.json to compare")
    parser.add_argument("--after-judge", type=str, default=None, help="with --after-only: path to an llm_judge_results.json to compare")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args)))
