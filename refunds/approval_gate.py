"""
Persistence for PendingAction records -- the HITL approval queue. Two backends behind
one interface (save/get/get_active_for_resource/all_pending/transition), selected by
HITL_STORE_BACKEND so refund_node (harness/graph.py), hitl_cli.py, and execute_refund's
revalidation don't know or care which one is live.

JsonFileStore: local dev. One JSON file, an object keyed by action_id -- matches the
instructor's hitl_cli.py contract exactly (see refunds/schemas.py's own docstring).
DynamoDBStore: AWS deployment. Table `ordercare-hitl-approvals`, partition key
action_id, one PendingAction per item.

No separate Postgres refund_tickets table -- this store IS the durable record,
including duplicate detection (get_active_for_resource). See refunds/schemas.py.

transition() is the one operation not spelled out by the instructor's interface --
added here specifically to close the concurrency bug the design doc calls out (two
reviewers approving the same PendingAction at once). Same contract on both backends:
an atomic compare-and-set that returns False (never raises) when expected_status
doesn't match current state, so callers treat False as "someone else already
resolved this," not a crash.
"""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

from refunds.schemas import ApprovalStatus, PendingAction

_ACTIVE_STATUSES = {ApprovalStatus.PENDING.value, ApprovalStatus.APPROVED.value}


class ApprovalStore(ABC):
    @abstractmethod
    def save(self, action: PendingAction) -> None: ...

    @abstractmethod
    def get(self, action_id: str) -> PendingAction | None: ...

    @abstractmethod
    def get_active_for_resource(self, order_id: str) -> PendingAction | None:
        """Duplicate-detection: an existing action on this order_id still in
        PENDING or APPROVED (not yet EXECUTED/REJECTED/EXPIRED). Correlates on
        order_id, never on ticket/session id -- two separate support tickets about
        the same order, weeks apart, must still find each other."""

    @abstractmethod
    def all_pending(self) -> list[PendingAction]: ...

    @abstractmethod
    def transition(
        self,
        action_id: str,
        expected_status: ApprovalStatus,
        new_status: ApprovalStatus,
        resolved_by: str | None = None,
    ) -> bool:
        """Atomic compare-and-set. False (never an exception) if action_id doesn't
        exist or its current status != expected_status."""


class JsonFileStore(ApprovalStore):
    """fcntl-locked read-modify-write around one JSON file -- the file IS the lock
    boundary (exclusive flock held for the full read+mutate+write), so two processes
    (e.g. two hitl_cli.py invocations) racing an approve/reject on the same action_id
    can't both see PENDING and both win; the second's transition() sees the first's
    write and returns False."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            self._path.write_text("{}")

    def _locked_read_modify_write(self, mutate):
        """mutate(data: dict) -> return value. Runs under an exclusive lock held for
        the file's full open->read->mutate->write->close lifetime."""
        import fcntl

        with open(self._path, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                data = json.loads(f.read() or "{}")
                result = mutate(data)
                f.seek(0)
                f.truncate()
                f.write(json.dumps(data, indent=2))
                return result
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _read(self) -> dict:
        return json.loads(self._path.read_text() or "{}")

    def save(self, action: PendingAction) -> None:
        item = json.loads(action.model_dump_json())
        self._locked_read_modify_write(lambda data: data.__setitem__(action.action_id, item))

    def get(self, action_id: str) -> PendingAction | None:
        raw = self._read().get(action_id)
        return PendingAction.model_validate(raw) if raw else None

    def get_active_for_resource(self, order_id: str) -> PendingAction | None:
        for raw in self._read().values():
            if raw["order_id"] == order_id and raw["status"] in _ACTIVE_STATUSES:
                return PendingAction.model_validate(raw)
        return None

    def all_pending(self) -> list[PendingAction]:
        return [
            PendingAction.model_validate(raw)
            for raw in self._read().values()
            if raw["status"] == ApprovalStatus.PENDING.value
        ]

    def transition(
        self,
        action_id: str,
        expected_status: ApprovalStatus,
        new_status: ApprovalStatus,
        resolved_by: str | None = None,
    ) -> bool:
        def mutate(data: dict) -> bool:
            raw = data.get(action_id)
            if raw is None or raw["status"] != expected_status.value:
                return False
            raw["status"] = new_status.value
            raw["resolved_by"] = resolved_by
            raw["resolved_at"] = datetime.now(timezone.utc).isoformat()
            data[action_id] = raw
            return True

        return self._locked_read_modify_write(mutate)


class DynamoDBStore(ApprovalStore):
    """get_active_for_resource/all_pending use a table scan, not a query -- no GSI on
    order_id or status exists. Fine at this app's scale (a handful of open refunds at
    a time); a GSI would be the real fix at production volume, not built here (see
    log/loophole.md). transition()'s atomicity comes from DynamoDB's own
    ConditionExpression, not from anything in this class."""

    def __init__(self, table_name: str, region: str | None = None) -> None:
        import boto3

        self._table = boto3.resource("dynamodb", region_name=region).Table(table_name)

    def save(self, action: PendingAction) -> None:
        self._table.put_item(Item=json.loads(action.model_dump_json()))

    def get(self, action_id: str) -> PendingAction | None:
        item = self._table.get_item(Key={"action_id": action_id}).get("Item")
        return PendingAction.model_validate(item) if item else None

    def get_active_for_resource(self, order_id: str) -> PendingAction | None:
        resp = self._table.scan(
            FilterExpression="order_id = :oid AND (#s = :pending OR #s = :approved)",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":oid": order_id,
                ":pending": ApprovalStatus.PENDING.value,
                ":approved": ApprovalStatus.APPROVED.value,
            },
        )
        items = resp.get("Items", [])
        return PendingAction.model_validate(items[0]) if items else None

    def all_pending(self) -> list[PendingAction]:
        resp = self._table.scan(
            FilterExpression="#s = :pending",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":pending": ApprovalStatus.PENDING.value},
        )
        return [PendingAction.model_validate(item) for item in resp.get("Items", [])]

    def transition(
        self,
        action_id: str,
        expected_status: ApprovalStatus,
        new_status: ApprovalStatus,
        resolved_by: str | None = None,
    ) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._table.update_item(
                Key={"action_id": action_id},
                UpdateExpression="SET #s = :new, resolved_by = :by, resolved_at = :at",
                ConditionExpression="#s = :expected",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={
                    ":new": new_status.value,
                    ":expected": expected_status.value,
                    ":by": resolved_by,
                    ":at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise


_store: ApprovalStore | None = None


def get_store() -> ApprovalStore:
    """HITL_STORE_BACKEND=dynamodb in AWS (task-def env var), unset/json locally.
    Cached -- one store instance per process, matching harness/llm_provider.py's
    get_llm_client() pattern."""
    global _store
    if _store is None:
        backend = os.environ.get("HITL_STORE_BACKEND", "json").lower()
        if backend == "dynamodb":
            _store = DynamoDBStore(
                table_name=os.environ.get("HITL_DYNAMODB_TABLE", "ordercare-hitl-approvals")
            )
        else:
            _store = JsonFileStore(os.environ.get("HITL_JSON_PATH", "pending_actions.json"))
    return _store
