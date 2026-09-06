from __future__ import annotations

import hashlib
import json
import uuid
from datetime import timedelta
from typing import Any

from april_common.audit import AuditLogger
from april_common.errors import ApprovalRequiredError, PermissionDeniedError
from april_common.time import parse_utc_iso, utc_now, utc_now_iso
from services.memory.database import Database
from services.memory.schemas import ApprovalRecord
from services.permissions.schemas import ApprovalRequest, ApprovalResponse


def canonical_hash(tool: str, args: dict[str, Any], metadata: dict[str, Any] | None = None) -> str:
    payload = json.dumps(
        {"tool": tool, "args": args, "metadata": metadata or {}},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def legacy_canonical_hash(tool: str, args: dict[str, Any]) -> str:
    payload = json.dumps({"tool": tool, "args": args}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_args_hash(args: dict[str, Any]) -> str:
    payload = json.dumps(args, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ApprovalStore:
    def __init__(self, database: Database, audit: AuditLogger, *, expiry_seconds: int) -> None:
        self.database = database
        self.audit = audit
        self.expiry_seconds = expiry_seconds

    async def create(
        self, request: ApprovalRequest, *, actor: str, request_id: str
    ) -> ApprovalResponse:
        approval_id = str(uuid.uuid4())
        expires_at = (
            (utc_now() + timedelta(seconds=self.expiry_seconds)).isoformat().replace("+00:00", "Z")
        )
        metadata = dict(request.metadata)
        if metadata:
            metadata["approval_id"] = approval_id
            metadata["approval_expires_at"] = expires_at
        digest = canonical_hash(request.tool, request.args, metadata)
        audit_error: Exception | None = None
        async with self.database.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO approvals(
                    id, tool, args_json, agent, canonical_hash, metadata_json,
                    permission_level, risk_level, status, expires_at, created_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    approval_id,
                    request.tool,
                    json.dumps(request.args, sort_keys=True),
                    request.agent,
                    digest,
                    json.dumps(metadata, sort_keys=True),
                    request.permission_level,
                    request.risk_level,
                    expires_at,
                    utc_now_iso(),
                ),
            )
            try:
                self.audit.write(
                    {
                        "actor": actor,
                        "request_id": request_id,
                        "event_type": "approval_created",
                        "tool": request.tool,
                        "arguments": request.args,
                        "agent": request.agent,
                        "permission_level": request.permission_level,
                        "risk": request.risk_level,
                        "metadata": metadata,
                        "approval_id": approval_id,
                        "outcome": "pending",
                    }
                )
            except Exception as exc:
                audit_error = exc
                await self._mark_audit_failed(
                    conn, approval_id=approval_id, error=exc, prior_status="pending"
                )
        if audit_error is not None:
            raise audit_error
        return ApprovalResponse(
            approval_id=approval_id,
            tool=request.tool,
            args=request.args,
            agent=request.agent,
            permission_level=request.permission_level,
            risk_level=request.risk_level,
            affected_paths=request.affected_paths,
            expected_side_effects=request.expected_side_effects,
            metadata=metadata,
            expires_at=expires_at,
        )

    async def list_pending(self) -> list[ApprovalRecord]:
        rows = await self.database.fetchall(
            "SELECT * FROM approvals WHERE status = 'pending' ORDER BY created_at ASC"
        )
        return [self._record_from_row(row) for row in rows]

    async def get(self, approval_id: str) -> ApprovalRecord:
        row = await self.database.fetchone("SELECT * FROM approvals WHERE id = ?", (approval_id,))
        if row is None:
            raise PermissionDeniedError("Approval does not exist.")
        return self._record_from_row(row)

    async def validate_exact(
        self,
        *,
        approval_id: str,
        tool: str,
        args: dict[str, Any],
    ) -> ApprovalRecord:
        """Check an approved, unexpired approval without consuming it."""
        record = await self.get(approval_id)
        if record.status != "approved":
            raise PermissionDeniedError("Approval is not approved.", {"status": record.status})
        try:
            unexpired = parse_utc_iso(record.expires_at) >= utc_now()
        except ValueError:
            unexpired = False
        if not unexpired:
            raise PermissionDeniedError("Approval has expired.")
        expected_hash = canonical_hash(tool, args, record.metadata)
        legacy_hash = legacy_canonical_hash(tool, args)
        hash_matches = record.canonical_hash == expected_hash or (
            not record.metadata and record.canonical_hash == legacy_hash
        )
        if record.tool != tool or not hash_matches:
            raise PermissionDeniedError("Approval arguments changed.")
        return record

    async def approve_exact(
        self,
        *,
        approval_id: str,
        tool: str,
        args: dict[str, Any],
        actor: str,
        request_id: str,
    ) -> ApprovalRecord:
        expired = False
        audit_error: Exception | None = None
        async with self.database.transaction() as conn:
            cursor = await conn.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,))
            row = await cursor.fetchone()
            if row is None:
                raise PermissionDeniedError("Approval does not exist.")
            record = self._record_from_row(row)
            now = utc_now()
            if record.status != "pending":
                raise PermissionDeniedError("Approval is not pending.", {"status": record.status})
            if parse_utc_iso(record.expires_at) < now:
                cursor = await conn.execute(
                    """
                    UPDATE approvals
                    SET status = 'expired'
                    WHERE id = ? AND status = 'pending'
                    """,
                    (approval_id,),
                )
                expired = cursor.rowcount == 1
                if not expired:
                    raise PermissionDeniedError("Approval is not pending.")
                await self._mark_suspended_terminal(
                    conn,
                    approval_id=approval_id,
                    suspended_status="expired",
                    run_status="expired",
                )
            else:
                expected_hash = canonical_hash(tool, args, record.metadata)
                legacy_hash = legacy_canonical_hash(tool, args)
                hash_matches = record.canonical_hash == expected_hash or (
                    not record.metadata and record.canonical_hash == legacy_hash
                )
                if record.tool != tool or not hash_matches:
                    raise PermissionDeniedError("Approval arguments changed.")
                cursor = await conn.execute(
                    """
                    UPDATE approvals
                    SET status = 'approved'
                    WHERE id = ? AND status = 'pending'
                    """,
                    (approval_id,),
                )
                if cursor.rowcount != 1:
                    raise PermissionDeniedError("Approval is not pending.")
                try:
                    self.audit.write(
                        {
                            "actor": actor,
                            "request_id": request_id,
                            "event_type": "approval_approved",
                            "tool": tool,
                            "arguments": args,
                            "agent": record.agent,
                            "permission_level": record.permission_level,
                            "risk": record.risk_level,
                            "metadata": record.metadata,
                            "approval_id": approval_id,
                            "outcome": "approved",
                        }
                    )
                except Exception as exc:
                    audit_error = exc
                    await self._mark_audit_failed(
                        conn, approval_id=approval_id, error=exc, prior_status="approved"
                    )
        if audit_error is not None:
            raise audit_error
        if expired:
            self._write_audit_event(
                record=record,
                actor=actor,
                request_id=request_id,
                event_type="approval_expired",
                outcome="expired",
            )
            raise PermissionDeniedError("Approval has expired.")
        return record.model_copy(update={"status": "approved"})

    async def consume(
        self,
        *,
        approval_id: str,
        result: dict[str, Any],
        actor: str,
        request_id: str,
    ) -> None:
        audit_error: Exception | None = None
        async with self.database.transaction() as conn:
            cursor = await conn.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,))
            row = await cursor.fetchone()
            if row is None:
                raise PermissionDeniedError("Approval does not exist.")
            record = self._record_from_row(row)
            if record.status != "approved":
                raise PermissionDeniedError("Approval has not been approved.")
            consumed_at = utc_now_iso()
            consumed = await self.consume_in_transaction(
                conn,
                approval_id=approval_id,
                result=result,
                consumed_at=consumed_at,
            )
            if not consumed:
                raise PermissionDeniedError("Approval has not been approved.")
            try:
                self.audit.write(
                    self._approval_event(record, actor, request_id, "approval_consumed", "consumed")
                )
            except Exception as exc:
                audit_error = exc
                await self._mark_audit_failed(
                    conn, approval_id=approval_id, error=exc, prior_status="consumed"
                )
        if audit_error is not None:
            raise audit_error

    async def consume_exact(
        self,
        *,
        approval_id: str,
        tool: str,
        args: dict[str, Any],
        result: dict[str, Any],
        actor: str,
        request_id: str,
    ) -> ApprovalRecord:
        """Validate and consume one approved, unexpired exact operation.

        This is used by sensitive CLI operations that do not execute through
        ``ToolExecutionService`` but still need the same one-time approval
        guarantees. The operation binding is checked inside the transaction so
        a caller cannot validate one action and consume another.
        """
        audit_error: Exception | None = None
        async with self.database.transaction() as conn:
            cursor = await conn.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,))
            row = await cursor.fetchone()
            if row is None:
                raise PermissionDeniedError("Approval does not exist.")
            record = self._record_from_row(row)
            if record.status != "approved":
                raise PermissionDeniedError("Approval is not approved.", {"status": record.status})
            try:
                unexpired = parse_utc_iso(record.expires_at) >= utc_now()
            except ValueError:
                unexpired = False
            if not unexpired:
                raise PermissionDeniedError("Approval has expired.")
            expected_hash = canonical_hash(tool, args, record.metadata)
            legacy_hash = legacy_canonical_hash(tool, args)
            hash_matches = record.canonical_hash == expected_hash or (
                not record.metadata and record.canonical_hash == legacy_hash
            )
            if record.tool != tool or not hash_matches:
                raise PermissionDeniedError("Approval arguments changed.")
            consumed_at = utc_now_iso()
            if not await self.consume_in_transaction(
                conn,
                approval_id=approval_id,
                result=result,
                consumed_at=consumed_at,
            ):
                raise PermissionDeniedError("Approval is no longer approved.")
            try:
                self.audit.write(
                    self._approval_event(record, actor, request_id, "approval_consumed", "consumed")
                )
            except Exception as exc:
                audit_error = exc
                await self._mark_audit_failed(
                    conn, approval_id=approval_id, error=exc, prior_status="consumed"
                )
        if audit_error is not None:
            raise audit_error
        return record.model_copy(update={"status": "consumed"})

    @classmethod
    async def consume_in_transaction(
        cls,
        conn: Any,
        *,
        approval_id: str,
        result: dict[str, Any],
        consumed_at: str,
    ) -> bool:
        """Consume an approved action inside the caller's SQLite transaction."""
        cursor = await conn.execute(
            """
            UPDATE approvals
            SET status = 'consumed', consumed_at = ?, result_json = ?
            WHERE id = ? AND status = 'approved'
            """,
            (consumed_at, json.dumps(result, sort_keys=True), approval_id),
        )
        if cursor.rowcount != 1:
            return False
        await cls._transition_suspended_after_consumption(
            conn,
            approval_id=approval_id,
            succeeded=result.get("ok") is True,
            transitioned_at=consumed_at,
        )
        return True

    async def deny(self, *, approval_id: str, actor: str, request_id: str) -> None:
        async with self.database.transaction() as conn:
            cursor = await conn.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,))
            row = await cursor.fetchone()
            if row is None:
                raise PermissionDeniedError("Approval does not exist.")
            record = self._record_from_row(row)
            if record.status != "pending":
                raise PermissionDeniedError("Approval is not pending.", {"status": record.status})
            cursor = await conn.execute(
                """
                UPDATE approvals
                SET status = 'denied'
                WHERE id = ? AND status = 'pending'
                """,
                (approval_id,),
            )
            if cursor.rowcount != 1:
                raise PermissionDeniedError("Approval is not pending.")
            await self._mark_suspended_terminal(
                conn,
                approval_id=approval_id,
                suspended_status="denied",
                run_status="denied",
            )
        self.audit.write(
            {
                "actor": actor,
                "request_id": request_id,
                "event_type": "approval_denied",
                "tool": record.tool,
                "arguments": record.args,
                "agent": record.agent,
                "permission_level": record.permission_level,
                "risk": record.risk_level,
                "metadata": record.metadata,
                "approval_id": approval_id,
                "outcome": "denied",
            }
        )

    async def expire_pending(self, *, approval_id: str, actor: str, request_id: str) -> None:
        async with self.database.transaction() as conn:
            cursor = await conn.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,))
            row = await cursor.fetchone()
            if row is None:
                raise PermissionDeniedError("Approval does not exist.")
            record = self._record_from_row(row)
            if record.status != "pending":
                raise PermissionDeniedError("Approval is not pending.", {"status": record.status})
            cursor = await conn.execute(
                """
                UPDATE approvals
                SET status = 'expired'
                WHERE id = ? AND status = 'pending'
                """,
                (approval_id,),
            )
            if cursor.rowcount != 1:
                raise PermissionDeniedError("Approval is not pending.")
            await self._mark_suspended_terminal(
                conn,
                approval_id=approval_id,
                suspended_status="expired",
                run_status="expired",
            )
        self.audit.write(
            {
                "actor": actor,
                "request_id": request_id,
                "event_type": "approval_expired",
                "tool": record.tool,
                "arguments": record.args,
                "agent": record.agent,
                "permission_level": record.permission_level,
                "risk": record.risk_level,
                "metadata": record.metadata,
                "approval_id": approval_id,
                "outcome": "expired",
            }
        )

    def require_approval(self, response: ApprovalResponse) -> None:
        raise ApprovalRequiredError(
            "This action requires approval.",
            {"approval": response.model_dump()},
        )

    def _write_audit_event(
        self,
        *,
        record: ApprovalRecord,
        actor: str,
        request_id: str,
        event_type: str,
        outcome: str,
    ) -> None:
        self.audit.write(
            {
                "actor": actor,
                "request_id": request_id,
                "event_type": event_type,
                "tool": record.tool,
                "arguments": record.args,
                "agent": record.agent,
                "permission_level": record.permission_level,
                "risk": record.risk_level,
                "metadata": record.metadata,
                "approval_id": record.id,
                "outcome": outcome,
            }
        )

    @staticmethod
    def _audit_failure(error: Exception) -> dict[str, str]:
        code = getattr(error, "code", None)
        return {
            "type": type(error).__name__,
            "code": str(code) if isinstance(code, str) else "audit_write_failed",
        }

    @classmethod
    async def _mark_audit_failed(
        cls,
        conn: Any,
        *,
        approval_id: str,
        error: Exception,
        prior_status: str,
    ) -> None:
        """Leave durable, non-authorizing evidence after an audit failure.

        This runs before the surrounding transaction commits.  The row is
        intentionally terminal, so a failed append can never be mistaken for
        a pending or approved transition by a later process.
        """
        evidence = json.dumps(
            {"audit_failure": cls._audit_failure(error), "prior_status": prior_status},
            sort_keys=True,
        )
        await conn.execute(
            "UPDATE approvals SET status = 'audit_failed', result_json = ? WHERE id = ?",
            (evidence, approval_id),
        )
        suspended = await conn.execute(
            "SELECT agent_run_id, status FROM suspended_agent_runs WHERE approval_id = ?",
            (approval_id,),
        )
        row = await suspended.fetchone()
        if row is not None and row["status"] in {"suspended", "resumed"}:
            await conn.execute(
                """
                UPDATE suspended_agent_runs
                SET status = 'failed', completed_at = ?
                WHERE approval_id = ?
                """,
                (utc_now_iso(), approval_id),
            )
            await conn.execute(
                "UPDATE agent_runs SET status = 'failed', completed_at = ? WHERE id = ?",
                (utc_now_iso(), row["agent_run_id"]),
            )

    @staticmethod
    def _approval_event(
        record: ApprovalRecord,
        actor: str,
        request_id: str,
        event_type: str,
        outcome: str,
    ) -> dict[str, Any]:
        return {
            "actor": actor,
            "request_id": request_id,
            "event_type": event_type,
            "tool": record.tool,
            "arguments": record.args,
            "agent": record.agent,
            "permission_level": record.permission_level,
            "risk": record.risk_level,
            "metadata": record.metadata,
            "approval_id": record.id,
            "outcome": outcome,
        }

    @staticmethod
    async def _mark_suspended_terminal(
        conn: Any,
        *,
        approval_id: str,
        suspended_status: str,
        run_status: str,
    ) -> None:
        cursor = await conn.execute(
            "SELECT agent_run_id FROM suspended_agent_runs WHERE approval_id = ?",
            (approval_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return
        now = utc_now_iso()
        cursor = await conn.execute(
            """
            UPDATE suspended_agent_runs
            SET status = ?, completed_at = ?
            WHERE approval_id = ? AND status = 'suspended'
            """,
            (suspended_status, now, approval_id),
        )
        if cursor.rowcount != 1:
            raise PermissionDeniedError("Suspended agent run is not pending.")
        cursor = await conn.execute(
            """
            UPDATE agent_runs
            SET status = ?, completed_at = ?
            WHERE id = ? AND status = 'suspended'
            """,
            (run_status, now, row["agent_run_id"]),
        )
        if cursor.rowcount != 1:
            raise PermissionDeniedError("Agent run is not suspended.")

    @staticmethod
    async def _transition_suspended_after_consumption(
        conn: Any,
        *,
        approval_id: str,
        succeeded: bool,
        transitioned_at: str,
    ) -> None:
        cursor = await conn.execute(
            "SELECT agent_run_id, status FROM suspended_agent_runs WHERE approval_id = ?",
            (approval_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return
        if row["status"] != "suspended":
            raise PermissionDeniedError(
                "Suspended agent run is not resumable.",
                {"status": row["status"]},
            )
        suspended_status = "resumed" if succeeded else "failed"
        run_status = "running" if succeeded else "failed"
        completed_at = None if succeeded else transitioned_at
        cursor = await conn.execute(
            """
            UPDATE suspended_agent_runs
            SET status = ?, resumed_at = ?, completed_at = ?
            WHERE approval_id = ? AND status = 'suspended'
            """,
            (
                suspended_status,
                transitioned_at if succeeded else None,
                completed_at,
                approval_id,
            ),
        )
        if cursor.rowcount != 1:
            raise PermissionDeniedError("Suspended agent run is not resumable.")
        cursor = await conn.execute(
            """
            UPDATE agent_runs
            SET status = ?, completed_at = ?
            WHERE id = ? AND status = 'suspended'
            """,
            (run_status, completed_at, row["agent_run_id"]),
        )
        if cursor.rowcount != 1:
            raise PermissionDeniedError("Agent run is not suspended.")

    def _record_from_row(self, row: Any) -> ApprovalRecord:
        data = dict(row)
        data["args"] = json.loads(data.pop("args_json"))
        data["metadata"] = json.loads(data.pop("metadata_json", "{}") or "{}")
        return ApprovalRecord.model_validate(data)
