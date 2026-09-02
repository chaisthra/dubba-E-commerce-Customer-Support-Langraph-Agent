"""
Pulls real traces from Langfuse and builds a DRAFT golden dataset for trajectory eval
(log/todos.md's "Golden dataset" + "Trajectory eval" items).

Built originally against client.api.trace.list() (langfuse SDK 4.14.1), which worked
against Langfuse Cloud with just a deprecation notice -- then broke outright (404)
once pointed at a self-hosted instance running in Langfuse v4 "events_only" mode
(log/DECISIONS.md), where that endpoint is fully disabled, not just deprecated.
Rewritten against client.api.observations.get_many(name="ticket-turn") instead -- the
non-deprecated v2 endpoint, works on both Cloud and self-hosted. A "trace" here is
just that root observation's trace_id; per-trace tool-call trajectory is a second
get_many(trace_id=...) call, same as before.

Only observation name="ticket-turn" (harness/graph.py's run_turn) counts as a candidate --
other trace names in this project (summarize-session, rolling-summarize, ...) are
sub-observations of a turn, not turns themselves; a Langfuse/LangGraph context-
propagation quirk in harness/summarizer.py's rolling summarization currently makes
those break out as separate top-level traces instead of nesting under their parent
turn (noted in log/DEV_LOG.md, not yet root-caused) -- filtering by name sidesteps it.

Session IDs from ad-hoc verification/testing (not real customer sessions) are
excluded by construction: harness/loop.py's new_session() always assigns a real
uuid4 session_id, so any trace whose session_id doesn't parse as a UUID is test
noise, not a real customer interaction, and gets dropped rather than hardcoding a
list of known test session IDs that would silently go stale.

expected_trajectory is deliberately left null for every row -- this script extracts
what actually happened (actual_trajectory), never what SHOULD have happened. A human
has to look at each row (query + actual_trajectory + final_reply + trace_url) and
decide correctness before it counts as ground truth; that's the whole point of
keeping this a "draft" file distinct from the eventual golden_dataset.json.
"""

import argparse
import json
import os
import uuid
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langfuse import Langfuse  # noqa: E402

TARGET_TYPES = ("order_status", "refund_request", "delivery_issue", "subscription_account")
MIN_TOTAL = 10
MIN_DISTINCT_TYPES = 3
PER_TYPE_CAP = 6


def _is_real_session(session_id: str | None) -> bool:
    if not session_id:
        return False
    try:
        uuid.UUID(session_id)
        return True
    except ValueError:
        return False


def fetch_ticket_turn_root_observations(client: Langfuse) -> list:
    observations = []
    cursor = None
    while True:
        resp = client.api.observations.get_many(
            name="ticket-turn", fields="core,basic,time,io,metadata", limit=100, cursor=cursor
        )
        observations.extend(resp.data)
        cursor = resp.meta.cursor
        if not cursor:
            break
    return [o for o in observations if _is_real_session(o.session_id)]


def _parse_io(value):
    """observations.get_many() returns TOOL input/output as JSON-encoded strings
    (our own instrumentation passes real dicts -- harness/graph.py's execute_tool_node
    -- but they come back through Langfuse's OTEL-attribute pipeline stringified).
    Parse back to a real object so the dataset holds structured data, not
    double-encoded JSON strings; fall back to the raw value if it's ever already
    parsed or genuinely isn't JSON."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def extract_trajectory(client: Langfuse, trace_id: str) -> list[dict]:
    obs = client.api.observations.get_many(trace_id=trace_id, fields="core,basic,time,io", limit=200)
    tool_calls = sorted(
        (o for o in obs.data if o.type == "TOOL" and o.name and o.name.startswith("execute-tool:")),
        key=lambda o: o.start_time,
    )
    return [
        {
            "tool": o.name.removeprefix("execute-tool:"),
            "arguments": _parse_io(o.input),
            "result": _parse_io(o.output),
        }
        for o in tool_calls
    ]


def build_rows(client: Langfuse, root_observations: list) -> list[dict]:
    rows = []
    for o in root_observations:
        categories = [c.strip() for c in (o.metadata or {}).get("categories", "").split(",") if c.strip()]
        rows.append(
            {
                "id": o.trace_id,
                "session_id": o.session_id,
                "timestamp": o.start_time.isoformat() if o.start_time else None,
                # Langfuse occasionally round-trips a bare numeric-looking string
                # (e.g. a user just typing "1002") back as a JSON number, not a
                # string -- normalize back to text, this is always a chat message.
                "query": str(o.input) if o.input is not None else "",
                "type": categories,
                "actual_trajectory": extract_trajectory(client, o.trace_id),
                "final_reply": o.output,
                "trace_url": f"{os.environ['LANGFUSE_BASE_URL'].rstrip('/')}/project/{o.project_id}/traces/{o.trace_id}",
                "expected_trajectory": None,
                "reviewed": False,
            }
        )
    return rows


def select_for_coverage(rows: list[dict]) -> list[dict]:
    """Picks a subset covering as many TARGET_TYPES as possible, up to PER_TYPE_CAP
    each, preferring rows with a non-empty actual_trajectory first (a golden
    trajectory dataset is most useful when it actually exercises tool calls), most
    recent first as a tiebreaker. A multi-category row counts toward every type it
    touches, so it can fill more than one type's quota."""
    by_type = defaultdict(list)
    for row in rows:
        for t in row["type"]:
            if t in TARGET_TYPES:
                by_type[t].append(row)
    for t in by_type:
        by_type[t].sort(key=lambda r: (len(r["actual_trajectory"]) == 0, r["timestamp"]), reverse=False)
        # non-empty trajectory first (False < True), then timestamp ascending == oldest
        # first within that tier -- reversed below to get newest-first per tier.
        by_type[t] = sorted(by_type[t], key=lambda r: (len(r["actual_trajectory"]) == 0, r["timestamp"] or ""))

    selected: dict[str, dict] = {}
    for t in TARGET_TYPES:
        for row in by_type.get(t, [])[:PER_TYPE_CAP]:
            selected[row["id"]] = row

    return sorted(selected.values(), key=lambda r: r["timestamp"] or "", reverse=True)


def print_summary(rows: list[dict]) -> None:
    print(f"\n{'#':<3} {'TYPE':<28} {'QUERY':<45} {'TRAJECTORY'}")
    print("-" * 130)
    for i, row in enumerate(rows, 1):
        type_str = ",".join(row["type"]) or "(none)"
        query_full = str(row["query"]) if row["query"] is not None else ""
        query_str = query_full[:42].replace("\n", " ")
        if len(query_full) > 42:
            query_str += "..."
        traj_str = " -> ".join(tc["tool"] for tc in row["actual_trajectory"]) or "(no tool calls)"
        print(f"{i:<3} {type_str:<28} {query_str:<45} {traj_str}")
    print("-" * 130)

    type_counts = defaultdict(int)
    for row in rows:
        for t in row["type"]:
            type_counts[t] += 1
    print(f"\n{len(rows)} rows total. Type coverage: {dict(type_counts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "golden_dataset_draft.json"))
    args = parser.parse_args()

    client = Langfuse()
    root_observations = fetch_ticket_turn_root_observations(client)
    print(f"Fetched {len(root_observations)} real ticket-turn traces (test/synthetic sessions excluded).")

    rows = build_rows(client, root_observations)
    selected = select_for_coverage(rows)

    distinct_types = {t for row in selected for t in row["type"] if t in TARGET_TYPES}
    if len(selected) < MIN_TOTAL or len(distinct_types) < MIN_DISTINCT_TYPES:
        print(
            f"\nWARNING: only got {len(selected)} rows across {len(distinct_types)} target types "
            f"(need >= {MIN_TOTAL} rows across >= {MIN_DISTINCT_TYPES} types). "
            "Not enough real trace history yet -- run more real conversations through "
            "python main.py, then re-run this script."
        )

    Path(args.output).write_text(json.dumps(selected, indent=2, default=str))
    print(f"\nWrote {len(selected)} draft rows to {args.output}")
    print("expected_trajectory is null on every row -- review actual_trajectory against")
    print("trace_url for each, then fill in expected_trajectory before this becomes the")
    print("real golden_dataset.json.")

    print_summary(selected)


if __name__ == "__main__":
    main()
