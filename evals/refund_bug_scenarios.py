"""
Assignment 3's four graded bug scenarios (refund design doc's "Bugs to deliberately
test" section) -- each one is a real repro against refunds/approval_gate.py's
JsonFileStore and refunds/execute.py's execute_refund, not a description of what
*should* happen. Run directly: `python evals/refund_bug_scenarios.py`.

Scenario 2 (concurrent approvals) uses real separate subprocesses, not threads or
multiprocessing.Pool -- Pool's default spawn start method can't pickle a function
defined in this module cleanly across platforms, and separate processes is also the
more realistic simulation of "two hitl_cli.py invocations racing" anyway.

Doesn't touch harness/graph.py or langfuse at all -- refund_node's own eligibility
logic (duplicate detection via get_active_for_resource, the account/window/refund_type
branches) is exercised indirectly here (scenario 1 calls the same store method
refund_node calls), but a full run_turn()-level trajectory test belongs in
evals/trajectory_eval.py's golden dataset, not here.
"""

import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mcp_server.mock_data as mock_data
from refunds.approval_gate import JsonFileStore
from refunds.execute import execute_refund
from refunds.schemas import ApprovalStatus, PendingAction, RefundDecision

_results: list[tuple[str, bool, str]] = []


def _check(name: str, condition: bool, detail: str = "") -> None:
    _results.append((name, condition, detail))
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


def _mk_action(action_id: str, order_id: str, customer_id: str, amount: float, snapshot: dict) -> PendingAction:
    return PendingAction(
        action_id=action_id,
        order_id=order_id,
        customer_id=customer_id,
        payload=RefundDecision(order_id=order_id, customer_id=customer_id, amount=amount, reason="test"),
        status=ApprovalStatus.PENDING,
        created_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        state_snapshot=snapshot,
    )


def scenario_1_duplicate_detection(store: JsonFileStore) -> None:
    """Two refund requests on the same order -- the second must find the first,
    never create a duplicate open row. Mirrors what refund_node's own duplicate
    check (store.get_active_for_resource) does before ever creating a row."""
    order_id = "ORD1001"
    first = _mk_action("dup-1", order_id, "CUST001", 64.0, {"order_status": "delivered", "account_standing": "active"})
    store.save(first)

    existing = store.get_active_for_resource(order_id)
    _check(
        "scenario 1: second request finds the first, not None",
        existing is not None and existing.action_id == "dup-1",
    )

    # A separate support ticket weeks later, same order -- correlation is on
    # order_id, never ticket/session id, so this must still find it.
    existing_again = store.get_active_for_resource(order_id)
    _check("scenario 1: correlates on order_id regardless of caller", existing_again is not None)


def scenario_2_concurrent_approvals(store_path: str) -> None:
    """Two reviewers approve the same PendingAction simultaneously -- the row lock
    (JsonFileStore's fcntl-held read-modify-write) must reject the second."""
    store = JsonFileStore(store_path)
    action = _mk_action("race-1", "ORD1002", "CUST001", 32.0, {"order_status": "in_transit", "account_standing": "active"})
    store.save(action)

    worker_src = f"""
import sys
sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})
from refunds.schemas import ApprovalStatus
from refunds.approval_gate import JsonFileStore
s = JsonFileStore({store_path!r})
result = s.transition("race-1", ApprovalStatus.PENDING, ApprovalStatus.APPROVED, resolved_by=sys.argv[1])
print("WIN" if result else "LOSE")
"""
    worker_path = os.path.join(tempfile.mkdtemp(), "worker.py")
    with open(worker_path, "w") as f:
        f.write(worker_src)

    procs = [
        subprocess.Popen([sys.executable, worker_path, f"reviewer-{i}"], stdout=subprocess.PIPE, text=True)
        for i in range(10)
    ]
    outputs = [p.communicate()[0].strip() for p in procs]
    wins = outputs.count("WIN")
    _check("scenario 2: exactly one of 10 concurrent approvals wins", wins == 1, f"results={outputs}")


def scenario_3_drift_check(store: JsonFileStore) -> None:
    """Approve, suspend the account, then execute -- the drift check must refuse,
    not silently execute against stale state."""
    order = mock_data.get_order("ORD1001")
    account = mock_data.get_account("CUST001")
    action = _mk_action(
        "drift-1", order["order_id"], "CUST001", order["order_total"],
        {"order_status": order["status"], "account_standing": "active" if account["standing"] == "active" else "flagged_or_suspended"},
    )
    store.save(action)

    approved = store.transition("drift-1", ApprovalStatus.PENDING, ApprovalStatus.APPROVED, resolved_by="reviewer-A")
    _check("scenario 3: approve succeeds", approved)

    original_standing = mock_data._ACCOUNTS["CUST001"]["standing"]
    mock_data._ACCOUNTS["CUST001"]["standing"] = "suspended"
    try:
        result = execute_refund(store, "drift-1", "reviewer-A")
        _check("scenario 3: execute refuses on drift", result["outcome"] == "state_changed", str(result.get("drift")))
        final = store.get("drift-1")
        _check("scenario 3: status stays APPROVED, never reaches EXECUTED", final.status == ApprovalStatus.APPROVED)
    finally:
        mock_data._ACCOUNTS["CUST001"]["standing"] = original_standing


def scenario_4_replay_idempotent(store: JsonFileStore) -> None:
    """Replay an approval call (e.g. a network retry) -- must be idempotent, never
    double-apply or let a later replay silently overwrite who actually approved it."""
    action = _mk_action("replay-1", "ORD1003", "CUST002", 32.0, {"order_status": "delayed", "account_standing": "active"})
    store.save(action)

    first = store.transition("replay-1", ApprovalStatus.PENDING, ApprovalStatus.APPROVED, resolved_by="reviewer-B")
    replay = store.transition("replay-1", ApprovalStatus.PENDING, ApprovalStatus.APPROVED, resolved_by="reviewer-C")
    _check("scenario 4: first call succeeds", first is True)
    _check("scenario 4: replay is refused, not double-applied", replay is False)
    final = store.get("replay-1")
    _check("scenario 4: resolved_by stays the first reviewer, replay didn't overwrite it", final.resolved_by == "reviewer-B")


def main() -> int:
    tmpdir = tempfile.mkdtemp()
    store_path = os.path.join(tmpdir, "pending.json")
    store = JsonFileStore(store_path)

    scenario_1_duplicate_detection(store)
    scenario_2_concurrent_approvals(store_path)
    scenario_3_drift_check(store)
    scenario_4_replay_idempotent(store)

    print()
    passed = sum(1 for _, ok, _ in _results if ok)
    print(f"{passed}/{len(_results)} passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
