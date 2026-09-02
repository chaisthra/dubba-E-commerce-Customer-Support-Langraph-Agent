"""
Structured payloads for Assignment 3's refund node + HITL flow. Every payload crossing
a boundary (graph node -> approval store -> hitl_cli.py -> back into execute_refund) is
a model here, never parsed prose.

RefundDecision / PendingAction match an external contract (an instructor-provided
hitl_cli.py + approval_gate.py this app needs to interoperate with) -- field names and
shapes here are NOT ours to redesign, they're a fixed interface. See refunds/approval_gate.py
for the store that persists PendingAction records (JSON file locally, DynamoDB in AWS).

RefundEligibilityResult is ours, not part of that contract -- it's refund_node's own
internal eligibility computation (harness/graph.py), kept deliberately distinct from
RefundDecision (which is the request payload attached to a PendingAction) so the two
don't collide under one name. refund_node computes a RefundEligibilityResult first;
only when eligibility says "create a pending action" does a RefundDecision + PendingAction
actually get built and persisted.

No separate Postgres table for refund state -- PendingAction (this file) plus
approval_gate.py's store IS the durable record, including duplicate detection
(get_active_for_resource(order_id)). Two parallel stores tracking the identical
pending/approved/rejected/expired/executed lifecycle would just be two sources of
truth that can drift from each other.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    EXECUTED = "executed"


class RefundType(str, Enum):
    STANDARD = "standard"
    DAMAGED_IN_TRANSIT = "damaged_in_transit"
    NON_DELIVERY = "non_delivery"


class RefundOutcome(str, Enum):
    """What refund_node's eligibility computation actually decided -- distinct from
    ApprovalStatus, which is about the PendingAction's HITL lifecycle. A refund can
    reach CREATED (a PendingAction now exists, awaiting human review) without ever
    touching ApprovalStatus.APPROVED/REJECTED -- those only apply once a reviewer acts."""

    CREATED = "created"
    DUPLICATE_FOUND = "duplicate_found"
    WINDOW_CLOSED = "window_closed"
    ORDER_NOT_FOUND = "order_not_found"
    PERMISSION_DENIED = "permission_denied"
    ACCOUNT_BLOCKED = "account_blocked"


class StateSnapshot(BaseModel):
    """Captured once at request time (refund_node) and again at execution time
    (execute_refund's revalidation) -- the diff between the two is the drift check."""

    order_status: str
    shipped_date: str | None
    delivery_date: str | None
    account_standing: str
    refund_window_days_remaining: int | None
    captured_at: datetime


class RefundDecision(BaseModel):
    """The refund REQUEST payload -- what a PendingAction wraps. Fixed shape, from the
    instructor's hitl_cli.py contract: order_id, customer_id, amount, reason. NOT the
    outcome of refund_node's own eligibility reasoning -- see RefundEligibilityResult
    for that."""

    order_id: str
    customer_id: str
    amount: float = Field(gt=0)
    reason: str


class PendingAction(BaseModel):
    """The durable HITL record -- one row awaiting (or having received) a human
    decision. Fixed shape, from the instructor's hitl_cli.py/approval_gate.py contract.
    Persisted via refunds/approval_gate.py's store (JsonFileStore locally,
    DynamoDBStore in AWS), keyed by action_id."""

    action_id: str
    order_id: str
    customer_id: str
    action_type: str = "issue_refund"
    payload: RefundDecision
    status: ApprovalStatus
    created_at: datetime
    expires_at: datetime
    resolved_at: datetime | None = None
    resolved_by: str | None = None
    state_snapshot: dict


class RefundEligibilityResult(BaseModel):
    """refund_node's own eligibility computation -- deterministic, no LLM call. Not
    part of the instructor's contract; this is purely internal to how harness/graph.py
    decides whether to create a PendingAction at all. `customer_message` goes out to
    the customer VERBATIM (finalize_node does not rephrase it -- see harness/graph.py's
    finalize_node), since it's the one piece of text here a customer actually reads."""

    outcome: RefundOutcome
    action_id: str | None = None
    refund_type: RefundType | None = None
    customer_message: str
    internal_reason: str
    snapshot: StateSnapshot | None = None
