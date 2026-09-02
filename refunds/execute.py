"""
The re-validation step between "approved" and "executed" (refund design doc's
critical path). Approving a PendingAction is not the same as executing it -- real
state (account standing, order status) can change in between, so this re-fetches
live data and refuses if anything material drifted, rather than trusting the
snapshot captured back when refund_node first created the row.

Kept as its own importable module (not buried in hitl_cli.py's argparse flow)
specifically so the graded bug scenarios (log/loophole.md / the design doc's own
"Bugs to deliberately test" section) can call execute_refund() directly in a test,
not just through the CLI.
"""

from refunds.approval_gate import ApprovalStore
from refunds.schemas import ApprovalStatus


def _live_snapshot(order: dict | None, account: dict | None) -> dict:
    """Same derivation refund_node used for the ORIGINAL snapshot (StateSnapshot's
    account_standing field) -- comparing like-for-like, not a richer live value
    against a coarser stored one."""
    return {
        "order_status": order["status"] if order else "unknown",
        "account_standing": "active" if account and account["standing"] == "active" else "flagged_or_suspended",
    }


def execute_refund(store: ApprovalStore, action_id: str, reviewer_id: str) -> dict:
    """Returns {"outcome": ...} -- never raises for a normal refusal (not-approved,
    drift found, lost a concurrent race); those are expected outcomes for a
    human-facing CLI to report, not exceptional control flow."""
    from mcp_server.mock_data import get_account, get_order

    action = store.get(action_id)
    if action is None:
        return {"outcome": "not_found"}

    if action.status != ApprovalStatus.APPROVED:
        return {"outcome": "not_in_approved_state", "actual": action.status.value}

    order = get_order(action.order_id)
    account = get_account(action.customer_id)
    live = _live_snapshot(order, account)

    # Material fields only -- order_status and account_standing, per the design
    # doc. Cosmetic changes (anything else in state_snapshot) don't block.
    drift = {
        field: {"was": action.state_snapshot.get(field), "now": live[field]}
        for field in ("order_status", "account_standing")
        if action.state_snapshot.get(field) != live[field]
    }
    if drift:
        return {"outcome": "state_changed", "drift": drift}

    # transition() is the same compare-and-set used for approve/reject -- if
    # someone else (another reviewer, a retried request) already moved this
    # action off APPROVED between our get() above and this call, it returns
    # False rather than double-executing.
    ok = store.transition(action_id, ApprovalStatus.APPROVED, ApprovalStatus.EXECUTED, resolved_by=reviewer_id)
    if not ok:
        return {"outcome": "lost_race", "reason": "action was no longer APPROVED at transition time"}

    return {"outcome": "executed", "action_id": action_id}
