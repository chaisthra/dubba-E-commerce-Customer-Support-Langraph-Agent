"""
Combined eval score + regression gate (Assignment 2, section 2.4).

Combines BOTH eval signals into one number, on purpose -- this session's own runs
found real defects that only ONE of the two checks alone would have caught:
- lit_candle_return_denied and broken_candle_refund_ord1001 (see
  log/learnings/2026-08-29-llm-judge-run1-findings.md) both called exactly the
  right tools -- a rule-based trajectory check alone would score them PASS -- but
  gave wrong or contradicted-by-policy answers, which only the LLM judge caught.
- Conversely, the trajectory check is what catches a skipped or wrong-argument
  tool call (the delivery_taking_forever wrong-order case) at all -- a fluent,
  confident final answer can look fine to a judge even when the STEPS that
  produced it were wrong (the whole "confident wrong path" premise of section
  2.2). Neither signal subsumes the other; a gate built on only one reconstructs
  exactly the blind spot section 6's "common pitfall" warns about, one layer up.

score = TRAJECTORY_WEIGHT * (trajectory pass rate, 0-100) +
        JUDGE_WEIGHT * (LLM-judge average, 1-5 rescaled to 0-100)

Weighted 50/50 by default -- deliberately equal, not trajectory-dominant, for the
reason above. Adjustable via the constants below; if you change them, update the
Defensible justification in the README/PR (assignment section 3).

Regression gate: compares the freshly-computed score against evals/baseline.json
and exits non-zero if it dropped by more than REGRESSION_THRESHOLD_POINTS. Baseline
comparison logic lives HERE, in Python, not in the CI YAML -- per the assignment's
own build guide ("don't put the pass/fail logic in the YAML").

Usage:
    python evals/combined_score.py                    # run fresh, print score, no gate check
    python evals/combined_score.py --save-baseline     # run fresh, write evals/baseline.json (do this once, on a known-good version)
    python evals/combined_score.py --gate               # run fresh, compare to baseline, exit 1 on regression > threshold (what CI calls)
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import evals.llm_judge as llm_judge  # noqa: E402
import evals.trajectory_eval as trajectory_eval  # noqa: E402

TRAJECTORY_WEIGHT = 0.5
JUDGE_WEIGHT = 0.5
REGRESSION_THRESHOLD_POINTS = 10  # combined-score points allowed to drop from baseline before the gate fails
JUDGE_RUNS_FOR_GATE = 3  # fewer than the 5 used for manual inspection -- keeps CI time/cost bounded, still "more than once" per Session 3's own non-determinism warning


def _judge_score_pct(overall_average: float | None) -> float:
    """1-5 scale -> 0-100, so it's on the same footing as the trajectory pass rate."""
    if overall_average is None:
        return 0.0
    return max(0.0, min(100.0, (overall_average - 1) / 4 * 100))


async def compute_score(
    dataset_path: Path,
    trajectory_out: Path,
    judge_out: Path,
    judge_runs: int,
) -> dict:
    trajectory_results = await trajectory_eval.run_all(dataset_path)
    trajectory_score = trajectory_eval.report(trajectory_results)
    trajectory_out.write_text(
        json.dumps(
            {
                "score": trajectory_score,
                "passed": sum(1 for r in trajectory_results if r["passed"]),
                "total": len(trajectory_results),
                "results": trajectory_results,
            },
            indent=2,
        )
    )

    llm_judge.main(trajectory_out, dataset_path, judge_out, judge_runs)
    judge_data = json.loads(judge_out.read_text())

    trajectory_pct = trajectory_score * 100
    judge_pct = _judge_score_pct(judge_data["overall_average"])
    combined = round(TRAJECTORY_WEIGHT * trajectory_pct + JUDGE_WEIGHT * judge_pct, 2)

    return {
        "combined_score": combined,
        "trajectory_score_pct": round(trajectory_pct, 2),
        "trajectory_passed": sum(1 for r in trajectory_results if r["passed"]),
        "trajectory_total": len(trajectory_results),
        "judge_score_pct": round(judge_pct, 2),
        "judge_average": judge_data["overall_average"],
        "judge_runs_per_ticket": judge_runs,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _print_summary(label: str, s: dict) -> None:
    print(
        f"{label}: combined={s['combined_score']} "
        f"(trajectory={s['trajectory_score_pct']}% [{s['trajectory_passed']}/{s['trajectory_total']}], "
        f"judge={s['judge_score_pct']}% [avg {s['judge_average']}/5, {s['judge_runs_per_ticket']} runs/ticket])"
    )


async def main(
    dataset_path: Path,
    trajectory_out: Path,
    judge_out: Path,
    baseline_path: Path,
    judge_runs: int,
    save_baseline: bool,
    gate: bool,
) -> int:
    current = await compute_score(dataset_path, trajectory_out, judge_out, judge_runs)
    _print_summary("Current", current)

    if save_baseline:
        baseline_path.write_text(json.dumps(current, indent=2))
        print(f"\nSaved as new baseline: {baseline_path}")
        return 0

    if not gate:
        return 0

    if not baseline_path.exists():
        print(f"\nNo baseline found at {baseline_path} -- run with --save-baseline first. Gate cannot run without one.")
        return 1

    baseline = json.loads(baseline_path.read_text())
    _print_summary("Baseline", baseline)

    drop = round(baseline["combined_score"] - current["combined_score"], 2)
    print(f"\nDrop from baseline: {drop} points (threshold: {REGRESSION_THRESHOLD_POINTS})")

    if drop > REGRESSION_THRESHOLD_POINTS:
        print(f"GATE FAILED: score dropped {drop} points, exceeding the {REGRESSION_THRESHOLD_POINTS}-point threshold.")
        return 1

    print("GATE PASSED.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined trajectory+judge eval score, with baseline regression gate")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "evals" / "golden_dataset.json")
    parser.add_argument("--trajectory-out", type=Path, default=PROJECT_ROOT / "evals" / "trajectory_eval_results.json")
    parser.add_argument("--judge-out", type=Path, default=PROJECT_ROOT / "evals" / "llm_judge_results.json")
    parser.add_argument("--baseline", type=Path, default=PROJECT_ROOT / "evals" / "baseline.json")
    parser.add_argument("--judge-runs", type=int, default=JUDGE_RUNS_FOR_GATE)
    parser.add_argument("--save-baseline", action="store_true", help="write current score as the new baseline instead of gating")
    parser.add_argument("--gate", action="store_true", help="compare to baseline and exit non-zero on regression (what CI runs)")
    args = parser.parse_args()

    sys.exit(
        asyncio.run(
            main(
                args.dataset,
                args.trajectory_out,
                args.judge_out,
                args.baseline,
                args.judge_runs,
                args.save_baseline,
                args.gate,
            )
        )
    )
