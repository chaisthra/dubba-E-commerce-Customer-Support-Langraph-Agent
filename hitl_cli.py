"""
Reviewer-facing CLI for Assignment 3's refund HITL queue -- analogous to main.py
being the customer-facing entry point, this is the human-reviewer-facing one.
Talks to the same refunds.approval_gate store refund_node (harness/graph.py) writes
to -- JSON file locally, DynamoDB in AWS, selected by HITL_STORE_BACKEND (see
refunds/approval_gate.py's get_store()).

Three actions, deliberately separate (not "approve" auto-executing) -- the design
doc's own graded bug scenario #3 (approve, suspend the account, then execute, drift
check must refuse) only makes sense if approval and execution are two distinct
steps with a real window between them for state to change:

  list                          -- show every PENDING action
  approve <action_id> --reviewer ID
  reject  <action_id> --reviewer ID
  execute <action_id> --reviewer ID   -- re-validates against live state, see
                                         refunds/execute.py
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

from refunds.approval_gate import get_store  # noqa: E402  (must follow load_dotenv())
from refunds.execute import execute_refund  # noqa: E402
from refunds.schemas import ApprovalStatus  # noqa: E402


def cmd_list(args: argparse.Namespace) -> None:
    store = get_store()
    pending = store.all_pending()
    if not pending:
        print("No pending refund requests.")
        return
    for action in pending:
        print(
            f"{action.action_id}  order={action.order_id}  customer={action.customer_id}  "
            f"amount=${action.payload.amount:.2f}  expires={action.expires_at.isoformat()}\n"
            f"  reason: {action.payload.reason}"
        )


def cmd_approve(args: argparse.Namespace) -> None:
    store = get_store()
    ok = store.transition(args.action_id, ApprovalStatus.PENDING, ApprovalStatus.APPROVED, resolved_by=args.reviewer)
    if not ok:
        print(f"Could not approve {args.action_id} -- it's no longer PENDING (already resolved by someone else?).")
        return
    print(f"Approved {args.action_id}. Run `execute {args.action_id}` to actually process the refund.")


def cmd_reject(args: argparse.Namespace) -> None:
    store = get_store()
    ok = store.transition(args.action_id, ApprovalStatus.PENDING, ApprovalStatus.REJECTED, resolved_by=args.reviewer)
    if not ok:
        print(f"Could not reject {args.action_id} -- it's no longer PENDING (already resolved by someone else?).")
        return
    print(f"Rejected {args.action_id}.")


def cmd_execute(args: argparse.Namespace) -> None:
    store = get_store()
    result = execute_refund(store, args.action_id, args.reviewer)
    outcome = result["outcome"]
    if outcome == "executed":
        print(f"Executed {args.action_id}.")
    elif outcome == "not_in_approved_state":
        print(f"Refused: {args.action_id} is {result['actual']}, not APPROVED.")
    elif outcome == "state_changed":
        print(f"Refused: state drifted since approval -- {result['drift']}")
    elif outcome == "lost_race":
        print(f"Refused: {result['reason']}")
    elif outcome == "not_found":
        print(f"No such action: {args.action_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dubba refund HITL review CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Show every PENDING refund request")

    for name, fn in (("approve", cmd_approve), ("reject", cmd_reject), ("execute", cmd_execute)):
        p = sub.add_parser(name)
        p.add_argument("action_id")
        p.add_argument("--reviewer", required=True, help="Reviewer identity, recorded on the action")
        p.set_defaults(func=fn)

    sub.choices["list"].set_defaults(func=cmd_list)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
