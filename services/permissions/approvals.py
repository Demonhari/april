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

    async def approve_exact(
        self,
        *,
        approval_id: str,
        tool: str,
        args: dict[str, Any],
        actor: str,
        request_id: str,
    ) -> ApprovalRecord:
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
                await conn.execute(
                    "UPDATE approvals SET status = 'expired' WHERE id = ?",
                    (approval_id,),
                )
                raise PermissionDeniedError("Approval has expired.")
            expected_hash = canonical_hash(tool, args, record.metadata)
            legacy_hash = legacy_canonical_hash(tool, args)
            hash_matches = record.canonical_hash == expected_hash or (
                not record.metadata and record.canonical_hash == legacy_hash
            )
            if record.tool != tool or not hash_matches:
                raise PermissionDeniedError("Approval arguments changed.")
            await conn.execute(
                "UPDATE approvals SET status = 'approved' WHERE id = ?",
                (approval_id,),
            )
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
        return record.model_copy(update={"status": "approved"})

    async def consume(
        self,
        *,
        approval_id: str,
        result: dict[str, Any],
        actor: str,
        request_id: str,
    ) -> None:
        async with self.database.transaction() as conn:
            cursor = await conn.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,))
            row = await cursor.fetchone()
            if row is None:
                raise PermissionDeniedError("Approval does not exist.")
            record = self._record_from_row(row)
            if record.status != "approved":
                raise PermissionDeniedError("Approval has not been approved.")
            await conn.execute(
                """
                UPDATE approvals
                SET status = 'consumed', consumed_at = ?, result_json = ?
                WHERE id = ?
                """,
                (utc_now_iso(), json.dumps(result, sort_keys=True), approval_id),
            )
        self.audit.write(
            {
                "actor": actor,
                "request_id": request_id,
                "event_type": "approval_consumed",
                "tool": record.tool,
                "arguments": record.args,
                "agent": record.agent,
                "permission_level": record.permission_level,
                "risk": record.risk_level,
                "metadata": record.metadata,
                "approval_id": approval_id,
                "outcome": "consumed",
            }
        )

    async def deny(self, *, approval_id: str, actor: str, request_id: str) -> None:
        async with self.database.transaction() as conn:
            cursor = await conn.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,))
            row = await cursor.fetchone()
            if row is None:
                raise PermissionDeniedError("Approval does not exist.")
            record = self._record_from_row(row)
            if record.status != "pending":
                raise PermissionDeniedError("Approval is not pending.", {"status": record.status})
            await conn.execute(
                "UPDATE approvals SET status = 'denied' WHERE id = ?", (approval_id,)
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

    def require_approval(self, response: ApprovalResponse) -> None:
        raise ApprovalRequiredError(
            "This action requires approval.",
            {"approval": response.model_dump()},
        )

    def _record_from_row(self, row: Any) -> ApprovalRecord:
        data = dict(row)
        data["args"] = json.loads(data.pop("args_json"))
        data["metadata"] = json.loads(data.pop("metadata_json", "{}") or "{}")
        return ApprovalRecord.model_validate(data)
