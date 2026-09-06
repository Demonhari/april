from __future__ import annotations

import builtins
import hashlib
import json
import uuid
from datetime import timedelta
from typing import Any

import aiosqlite

from april_common.audit import AuditLogger
from april_common.time import parse_utc_iso, utc_now, utc_now_iso
from services.jobs.registry import JobRegistry
from services.jobs.schemas import (
    DEFAULT_JOB_LIST_LIMIT,
    MAX_JOB_EVENT_CODE_CHARS,
    MAX_JOB_EVENTS_PER_JOB,
    MAX_JOB_LIST_LIMIT,
    MAX_JOB_PAYLOAD_BYTES,
    MAX_JOB_RESULT_BYTES,
    TERMINAL_JOB_STATUSES,
    ClaimedJob,
    JobEvent,
    JobRecord,
    JobStatus,
)
from services.memory.database import Database
from services.permissions.approvals import ApprovalStore, canonical_hash, legacy_canonical_hash

_FORBIDDEN_STRUCTURED_KEYS = frozenset(
    {
        "token",
        "api_token",
        "runtime_token",
        "authorization",
        "password",
        "secret",
        "environment",
        "env",
        "reasoning",
        "raw_audio",
        "stdout",
        "stderr",
    }
)


class JobStoreError(RuntimeError):
    pass


class JobNotFoundError(JobStoreError):
    pass


class JobTransitionError(JobStoreError):
    pass


class JobStore:
    def __init__(
        self, database: Database, registry: JobRegistry, audit: AuditLogger | None = None
    ) -> None:
        self.database = database
        self.registry = registry
        self.audit = audit

    async def submit(
        self,
        *,
        job_type: str,
        payload: dict[str, Any],
        owner: str,
        conversation_id: str | None = None,
        project_id: str | None = None,
        job_id: str | None = None,
    ) -> JobRecord:
        definition = self.registry.require(job_type)
        if definition.approval_required:
            raise JobTransitionError("approval_required")
        _reject_forbidden_keys(payload)
        validated = definition.validate_payload(payload)
        payload_json = _safe_json(validated, MAX_JOB_PAYLOAD_BYTES, "payload")
        now = utc_now_iso()
        identifier = job_id or str(uuid.uuid4())
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        bounded_owner = _bounded_code(owner, "owner")
        bounded_conversation = _optional_bounded_identifier(conversation_id)
        bounded_project = _optional_bounded_identifier(project_id)
        async with self.database.transaction() as connection:
            await self._insert_job_tx(
                connection,
                identifier=identifier,
                job_type=job_type,
                payload_json=payload_json,
                payload_hash=payload_hash,
                owner=bounded_owner,
                conversation_id=bounded_conversation,
                project_id=bounded_project,
                maximum_attempts=definition.maximum_attempts,
                created_at=now,
            )
            await self._append_event_tx(
                connection,
                identifier,
                event_type="submitted",
                message_code="queued",
                progress_percent=0,
                created_at=now,
            )
        return await self.require(identifier)

    async def submit_with_exact_approval(
        self,
        *,
        job_type: str,
        payload: dict[str, Any],
        owner: str,
        approval_id: str,
        approval_tool: str,
        approval_args: dict[str, Any],
        conversation_id: str | None = None,
        project_id: str | None = None,
    ) -> tuple[JobRecord, bool]:
        """Atomically consume one exact approval and accept its durable job.

        The approval identifier is also the idempotency key. Replaying the same
        exact accepted action returns the original job; changed arguments fail
        closed.
        """
        definition = self.registry.require(job_type)
        if not definition.approval_required:
            raise JobTransitionError("approval_not_required")
        _reject_forbidden_keys(payload)
        validated = definition.validate_payload(payload)
        payload_json = _safe_json(validated, MAX_JOB_PAYLOAD_BYTES, "payload")
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        identifier = f"approved-{_bounded_identifier(approval_id)}"
        now = utc_now_iso()
        bounded_owner = _bounded_code(owner, "owner")
        bounded_conversation = _optional_bounded_identifier(conversation_id)
        bounded_project = _optional_bounded_identifier(project_id)
        created = False
        audit_error: Exception | None = None
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                "SELECT * FROM approvals WHERE id = ?",
                (_bounded_identifier(approval_id),),
            )
            approval = await cursor.fetchone()
            if approval is None:
                raise JobTransitionError("approval_not_found")
            try:
                stored_args = json.loads(str(approval["args_json"]))
                metadata = json.loads(str(approval["metadata_json"] or "{}"))
            except json.JSONDecodeError as exc:
                raise JobTransitionError("approval_record_invalid") from exc
            if not isinstance(stored_args, dict) or not isinstance(metadata, dict):
                raise JobTransitionError("approval_record_invalid")
            expected = canonical_hash(approval_tool, approval_args, metadata)
            legacy = legacy_canonical_hash(approval_tool, approval_args)
            hash_matches = str(approval["canonical_hash"]) == expected or (
                not metadata and str(approval["canonical_hash"]) == legacy
            )
            if (
                str(approval["tool"]) != approval_tool
                or stored_args != approval_args
                or not hash_matches
                or int(approval["permission_level"]) != definition.permission_level
                or not _approval_scope_matches(
                    approval_args=approval_args,
                    metadata=metadata,
                    conversation_id=bounded_conversation,
                    project_id=bounded_project,
                )
            ):
                raise JobTransitionError("approval_action_mismatch")
            status = str(approval["status"])
            if status == "consumed":
                cursor = await connection.execute(
                    "SELECT * FROM background_jobs WHERE id = ?",
                    (identifier,),
                )
                row = await cursor.fetchone()
                if (
                    row is None
                    or str(row["job_type"]) != job_type
                    or str(row["payload_hash"]) != payload_hash
                    or str(row["owner"]) != bounded_owner
                    or row["conversation_id"] != bounded_conversation
                    or row["project_id"] != bounded_project
                ):
                    raise JobTransitionError("approval_replay_mismatch")
                return _record_from_row(row), False
            if status == "expired":
                raise JobTransitionError("approval_expired")
            if status != "approved":
                raise JobTransitionError("approval_not_approved")
            try:
                expired = parse_utc_iso(str(approval["expires_at"])) < utc_now()
            except ValueError as exc:
                raise JobTransitionError("approval_record_invalid") from exc
            if expired:
                raise JobTransitionError("approval_expired")
            expected_payload = definition.expected_approval_payload(
                tool=approval_tool,
                args=approval_args,
            )
            if validated != expected_payload:
                raise JobTransitionError("approval_action_mismatch")
            await self._insert_job_tx(
                connection,
                identifier=identifier,
                job_type=job_type,
                payload_json=payload_json,
                payload_hash=payload_hash,
                owner=bounded_owner,
                conversation_id=bounded_conversation,
                project_id=bounded_project,
                maximum_attempts=definition.maximum_attempts,
                created_at=now,
            )
            await self._append_event_tx(
                connection,
                identifier,
                event_type="submitted",
                message_code="queued",
                progress_percent=0,
                created_at=now,
            )
            result = json.dumps(
                {"ok": True, "job_id": identifier, "status": JobStatus.QUEUED.value},
                sort_keys=True,
            )
            consumed = await self._consume_approval_tx(
                connection,
                approval_id=approval_id,
                result_json=result,
                consumed_at=now,
            )
            if not consumed:
                raise JobTransitionError("approval_consumption_race")
            created = True
            cursor = await connection.execute(
                "SELECT * FROM background_jobs WHERE id = ?",
                (identifier,),
            )
            row = await cursor.fetchone()
            assert row is not None
            if self.audit is not None:
                try:
                    self.audit.write(
                        {
                            "actor": owner,
                            "request_id": f"job-submit-{approval_id}",
                            "event_type": "approval_consumed",
                            "tool": approval_tool,
                            "approval_id": approval_id,
                            "outcome": "consumed",
                            "job_id": identifier,
                        }
                    )
                except Exception as exc:
                    audit_error = exc
                    await ApprovalStore._mark_audit_failed(
                        connection,
                        approval_id=approval_id,
                        error=exc,
                        prior_status="consumed",
                    )
        if audit_error is not None:
            raise audit_error
        return _record_from_row(row), created

    async def _insert_job_tx(
        self,
        connection: aiosqlite.Connection,
        *,
        identifier: str,
        job_type: str,
        payload_json: str,
        payload_hash: str,
        owner: str,
        conversation_id: str | None,
        project_id: str | None,
        maximum_attempts: int,
        created_at: str,
    ) -> None:
        try:
            await connection.execute(
                """
                INSERT INTO background_jobs(
                    id, job_type, status, payload_json, payload_hash, owner,
                    conversation_id, project_id, progress_percent, attempt_count,
                    maximum_attempts, cancellation_requested, created_at, updated_at
                ) VALUES(?, ?, 'queued', ?, ?, ?, ?, ?, 0, 0, ?, 0, ?, ?)
                """,
                (
                    identifier,
                    job_type,
                    payload_json,
                    payload_hash,
                    owner,
                    conversation_id,
                    project_id,
                    maximum_attempts,
                    created_at,
                    created_at,
                ),
            )
        except aiosqlite.IntegrityError as exc:
            raise JobTransitionError("duplicate_job_id") from exc

    async def _consume_approval_tx(
        self,
        connection: aiosqlite.Connection,
        *,
        approval_id: str,
        result_json: str,
        consumed_at: str,
    ) -> bool:
        result = json.loads(result_json)
        assert isinstance(result, dict)
        return await ApprovalStore.consume_in_transaction(
            connection,
            approval_id=approval_id,
            result=result,
            consumed_at=consumed_at,
        )

    async def checkpoint(
        self,
        job_id: str,
        *,
        worker_id: str,
        result: dict[str, Any],
        progress_percent: int,
        progress_code: str,
    ) -> JobRecord:
        """Persist a bounded safe partial result while retaining the worker lease."""
        result_json = _safe_json(result, MAX_JOB_RESULT_BYTES, "result")
        now = utc_now_iso()
        async with self.database.transaction() as connection:
            updated = await connection.execute(
                """
                UPDATE background_jobs
                SET result_json = ?, progress_percent = ?, progress_code = ?, updated_at = ?
                WHERE id = ? AND worker_id = ? AND status IN ('running', 'cancelling')
                """,
                (
                    result_json,
                    _progress(progress_percent),
                    _bounded_code(progress_code, "progress_code"),
                    now,
                    _bounded_identifier(job_id),
                    _bounded_code(worker_id, "worker_id"),
                ),
            )
            if updated.rowcount != 1:
                raise JobTransitionError("lease_not_owned")
            await self._append_event_tx(
                connection,
                job_id,
                event_type="checkpoint",
                message_code=progress_code,
                progress_percent=progress_percent,
                created_at=now,
            )
        return await self.require(job_id)

    async def require(self, job_id: str, *, include_payload: bool = False) -> JobRecord:
        row = await self.database.fetchone(
            "SELECT * FROM background_jobs WHERE id = ?",
            (_bounded_identifier(job_id),),
        )
        if row is None:
            raise JobNotFoundError("job_not_found")
        if include_payload:
            return _claimed_from_row(row)
        return _record_from_row(row)

    async def list(
        self,
        *,
        owner: str | None = None,
        project_id: str | None = None,
        limit: int = DEFAULT_JOB_LIST_LIMIT,
        offset: int = 0,
    ) -> list[JobRecord]:
        if not 1 <= limit <= MAX_JOB_LIST_LIMIT or not 0 <= offset <= 10_000:
            raise ValueError("pagination_out_of_bounds")
        clauses: list[str] = []
        parameters: list[Any] = []
        if owner is not None:
            clauses.append("owner = ?")
            parameters.append(_bounded_code(owner, "owner"))
        if project_id is not None:
            clauses.append("project_id = ?")
            parameters.append(_bounded_identifier(project_id))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.extend([limit, offset])
        rows = await self.database.fetchall(
            f"""
            SELECT * FROM background_jobs
            {where}
            ORDER BY created_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            tuple(parameters),
        )
        return [_record_from_row(row) for row in rows]

    async def claim_next(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
    ) -> ClaimedJob | None:
        worker = _bounded_code(worker_id, "worker_id")
        if not 1 <= lease_seconds <= 300:
            raise ValueError("lease_out_of_bounds")
        now_value = utc_now()
        now = now_value.isoformat().replace("+00:00", "Z")
        expires = (now_value + timedelta(seconds=lease_seconds)).isoformat().replace("+00:00", "Z")
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                """
                SELECT id FROM background_jobs
                WHERE status = 'queued'
                  AND cancellation_requested = 0
                  AND attempt_count < maximum_attempts
                  AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                ORDER BY created_at, id
                LIMIT 1
                """,
                (now,),
            )
            candidate = await cursor.fetchone()
            if candidate is None:
                return None
            job_id = str(candidate["id"])
            updated = await connection.execute(
                """
                UPDATE background_jobs
                SET status = 'running',
                    worker_id = ?,
                    lease_acquired_at = ?,
                    lease_expires_at = ?,
                    heartbeat_at = ?,
                    started_at = COALESCE(started_at, ?),
                    updated_at = ?,
                    attempt_count = attempt_count + 1
                WHERE id = ?
                  AND status = 'queued'
                  AND cancellation_requested = 0
                  AND attempt_count < maximum_attempts
                  AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                """,
                (worker, now, expires, now, now, now, job_id, now),
            )
            if updated.rowcount != 1:
                return None
            await self._append_event_tx(
                connection,
                job_id,
                event_type="claimed",
                message_code="running",
                progress_percent=None,
                created_at=now,
            )
            cursor = await connection.execute(
                "SELECT * FROM background_jobs WHERE id = ?",
                (job_id,),
            )
            row = await cursor.fetchone()
            assert row is not None
            return _claimed_from_row(row)

    async def heartbeat(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_seconds: int,
        progress_percent: int | None = None,
        progress_code: str | None = None,
    ) -> JobRecord:
        if not 1 <= lease_seconds <= 300:
            raise ValueError("lease_out_of_bounds")
        progress = None if progress_percent is None else _progress(progress_percent)
        code = None if progress_code is None else _bounded_code(progress_code, "progress_code")
        now_value = utc_now()
        now = now_value.isoformat().replace("+00:00", "Z")
        expires = (now_value + timedelta(seconds=lease_seconds)).isoformat().replace("+00:00", "Z")
        assignments = [
            "heartbeat_at = ?",
            "lease_expires_at = ?",
            "updated_at = ?",
        ]
        parameters: list[Any] = [now, expires, now]
        if progress is not None:
            assignments.append("progress_percent = ?")
            parameters.append(progress)
        if code is not None:
            assignments.append("progress_code = ?")
            parameters.append(code)
        parameters.extend([_bounded_identifier(job_id), _bounded_code(worker_id, "worker_id")])
        async with self.database.transaction() as connection:
            updated = await connection.execute(
                f"""
                UPDATE background_jobs
                SET {", ".join(assignments)}
                WHERE id = ? AND worker_id = ? AND status IN ('running', 'cancelling')
                """,
                tuple(parameters),
            )
            if updated.rowcount != 1:
                raise JobTransitionError("lease_not_owned")
            if progress is not None or code is not None:
                await self._append_event_tx(
                    connection,
                    job_id,
                    event_type="progress",
                    message_code=code,
                    progress_percent=progress,
                    created_at=now,
                )
        return await self.require(job_id)

    async def request_cancel(self, job_id: str) -> tuple[JobRecord, bool]:
        identifier = _bounded_identifier(job_id)
        now = utc_now_iso()
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                "SELECT status FROM background_jobs WHERE id = ?",
                (identifier,),
            )
            row = await cursor.fetchone()
            if row is None:
                raise JobNotFoundError("job_not_found")
            status = JobStatus(str(row["status"]))
            if status in TERMINAL_JOB_STATUSES:
                return await self.require(identifier), True
            if status is JobStatus.QUEUED:
                await connection.execute(
                    """
                    UPDATE background_jobs
                    SET status = 'cancelled', cancellation_requested = 1,
                        completed_at = ?, updated_at = ?, progress_code = 'cancelled_before_start'
                    WHERE id = ? AND status = 'queued'
                    """,
                    (now, now, identifier),
                )
            else:
                await connection.execute(
                    """
                    UPDATE background_jobs
                    SET status = 'cancelling', cancellation_requested = 1,
                        updated_at = ?, progress_code = 'cancellation_requested'
                    WHERE id = ? AND status IN ('running', 'cancelling')
                    """,
                    (now, identifier),
                )
            await self._append_event_tx(
                connection,
                identifier,
                event_type="cancellation",
                message_code="cancellation_requested",
                progress_percent=None,
                created_at=now,
            )
        return await self.require(identifier), False

    async def cancellation_requested(self, job_id: str, worker_id: str) -> bool:
        row = await self.database.fetchone(
            """
            SELECT cancellation_requested FROM background_jobs
            WHERE id = ? AND worker_id = ? AND status IN ('running', 'cancelling')
            """,
            (_bounded_identifier(job_id), _bounded_code(worker_id, "worker_id")),
        )
        return bool(row and row["cancellation_requested"])

    async def finish(
        self,
        job_id: str,
        *,
        worker_id: str,
        status: JobStatus,
        result: dict[str, Any] | None = None,
        error_code: str | None = None,
    ) -> JobRecord:
        if status not in TERMINAL_JOB_STATUSES:
            raise ValueError("finish_requires_terminal_status")
        result_json = None if result is None else _safe_json(result, MAX_JOB_RESULT_BYTES, "result")
        safe_error = None if error_code is None else _bounded_code(error_code, "error_code")
        now = utc_now_iso()
        progress = 100 if status is JobStatus.SUCCEEDED else None
        async with self.database.transaction() as connection:
            updated = await connection.execute(
                """
                UPDATE background_jobs
                SET status = ?, result_json = ?, error_code = ?,
                    progress_percent = COALESCE(?, progress_percent),
                    progress_code = ?, completed_at = ?, updated_at = ?,
                    lease_expires_at = NULL, heartbeat_at = ?
                WHERE id = ? AND worker_id = ? AND status IN ('running', 'cancelling')
                """,
                (
                    status.value,
                    result_json,
                    safe_error,
                    progress,
                    status.value,
                    now,
                    now,
                    now,
                    _bounded_identifier(job_id),
                    _bounded_code(worker_id, "worker_id"),
                ),
            )
            if updated.rowcount != 1:
                raise JobTransitionError("terminal_transition_denied")
            await self._append_event_tx(
                connection,
                job_id,
                event_type="terminal",
                message_code=status.value,
                progress_percent=progress,
                created_at=now,
            )
        return await self.require(job_id)

    async def retry(self, job_id: str) -> tuple[JobRecord, bool]:
        identifier = _bounded_identifier(job_id)
        now = utc_now_iso()
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                """
                SELECT job_type, status, attempt_count, maximum_attempts
                FROM background_jobs WHERE id = ?
                """,
                (identifier,),
            )
            row = await cursor.fetchone()
            if row is None:
                raise JobNotFoundError("job_not_found")
            status = JobStatus(str(row["status"]))
            if status is JobStatus.QUEUED:
                return await self.require(identifier), True
            definition = self.registry.require(str(row["job_type"]))
            if (
                status not in {JobStatus.FAILED, JobStatus.INTERRUPTED, JobStatus.CANCELLED}
                or not definition.idempotent
                or int(row["attempt_count"]) >= int(row["maximum_attempts"])
            ):
                raise JobTransitionError("retry_not_eligible")
            await connection.execute(
                """
                UPDATE background_jobs
                SET status = 'queued', cancellation_requested = 0, worker_id = NULL,
                    lease_acquired_at = NULL, lease_expires_at = NULL, heartbeat_at = NULL,
                    completed_at = NULL,
                    result_json = CASE
                        WHEN job_type = 'model_setup_comparison' THEN result_json
                        ELSE NULL
                    END,
                    error_code = NULL,
                    progress_percent = 0, progress_code = 'retry_queued', updated_at = ?
                WHERE id = ? AND status IN ('failed', 'interrupted', 'cancelled')
                """,
                (now, identifier),
            )
            await self._append_event_tx(
                connection,
                identifier,
                event_type="retry",
                message_code="retry_queued",
                progress_percent=0,
                created_at=now,
            )
        return await self.require(identifier), False

    async def recover_expired_leases(self) -> builtins.list[JobRecord]:
        now = utc_now_iso()
        recovered: list[str] = []
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                """
                SELECT id, job_type, status, attempt_count, maximum_attempts
                FROM background_jobs
                WHERE status IN ('running', 'cancelling')
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at <= ?
                ORDER BY id
                """,
                (now,),
            )
            rows = await cursor.fetchall()
            for row in rows:
                identifier = str(row["id"])
                definition = self.registry.get(str(row["job_type"]))
                cancelling = str(row["status"]) == JobStatus.CANCELLING.value
                retryable = bool(
                    not cancelling
                    and definition is not None
                    and definition.idempotent
                    and definition.restart_safe
                    and int(row["attempt_count"]) < int(row["maximum_attempts"])
                )
                target = (
                    JobStatus.QUEUED
                    if retryable
                    else (JobStatus.CANCELLED if cancelling else JobStatus.INTERRUPTED)
                )
                await connection.execute(
                    """
                    UPDATE background_jobs
                    SET status = ?, worker_id = NULL, lease_acquired_at = NULL,
                        lease_expires_at = NULL, heartbeat_at = NULL, updated_at = ?,
                        completed_at = CASE WHEN ? = 'queued' THEN NULL ELSE ? END,
                        error_code = CASE WHEN ? = 'queued' THEN NULL ELSE 'lease_expired' END,
                        progress_code = ?
                    WHERE id = ? AND status IN ('running', 'cancelling')
                      AND lease_expires_at <= ?
                    """,
                    (
                        target.value,
                        now,
                        target.value,
                        now,
                        target.value,
                        "lease_expired_requeued" if retryable else "lease_expired",
                        identifier,
                        now,
                    ),
                )
                await self._append_event_tx(
                    connection,
                    identifier,
                    event_type="recovery",
                    message_code="lease_expired_requeued" if retryable else target.value,
                    progress_percent=None,
                    created_at=now,
                )
                recovered.append(identifier)
        return [await self.require(identifier) for identifier in recovered]

    async def events(self, job_id: str, *, limit: int = 100) -> builtins.list[JobEvent]:
        if not 1 <= limit <= MAX_JOB_EVENTS_PER_JOB:
            raise ValueError("event_limit_out_of_bounds")
        await self.require(job_id)
        rows = await self.database.fetchall(
            """
            SELECT * FROM background_job_events
            WHERE job_id = ? ORDER BY id DESC LIMIT ?
            """,
            (_bounded_identifier(job_id), limit),
        )
        return [
            JobEvent(
                id=int(row["id"]),
                job_id=str(row["job_id"]),
                event_type=str(row["event_type"]),
                message_code=row["message_code"],
                progress_percent=row["progress_percent"],
                created_at=str(row["created_at"]),
            )
            for row in reversed(rows)
        ]

    async def counts(self) -> dict[str, int]:
        rows = await self.database.fetchall(
            "SELECT status, COUNT(*) AS count FROM background_jobs GROUP BY status"
        )
        counts = {status.value: 0 for status in JobStatus}
        counts.update({str(row["status"]): int(row["count"]) for row in rows})
        expired = await self.database.fetchone(
            """
            SELECT COUNT(*) AS count FROM background_jobs
            WHERE status IN ('running', 'cancelling')
              AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?
            """,
            (utc_now_iso(),),
        )
        counts["expired_leases"] = int(expired["count"]) if expired else 0
        return counts

    async def _append_event_tx(
        self,
        connection: aiosqlite.Connection,
        job_id: str,
        *,
        event_type: str,
        message_code: str | None,
        progress_percent: int | None,
        created_at: str,
    ) -> None:
        event = _bounded_code(event_type, "event_type")
        message = None if message_code is None else _bounded_code(message_code, "message_code")
        await connection.execute(
            """
            INSERT INTO background_job_events(
                job_id, event_type, message_code, progress_percent, created_at
            ) VALUES(?, ?, ?, ?, ?)
            """,
            (job_id, event, message, progress_percent, created_at),
        )
        await connection.execute(
            """
            DELETE FROM background_job_events
            WHERE job_id = ? AND id NOT IN (
                SELECT id FROM background_job_events
                WHERE job_id = ? ORDER BY id DESC LIMIT ?
            )
            """,
            (job_id, job_id, MAX_JOB_EVENTS_PER_JOB),
        )


def _record_from_row(row: Any) -> JobRecord:
    result = json.loads(str(row["result_json"])) if row["result_json"] else None
    return JobRecord(
        id=str(row["id"]),
        job_type=str(row["job_type"]),
        status=JobStatus(str(row["status"])),
        payload_hash=str(row["payload_hash"]),
        owner=str(row["owner"]),
        conversation_id=row["conversation_id"],
        project_id=row["project_id"],
        progress_percent=int(row["progress_percent"]),
        progress_code=row["progress_code"],
        attempt_count=int(row["attempt_count"]),
        maximum_attempts=int(row["maximum_attempts"]),
        cancellation_requested=bool(row["cancellation_requested"]),
        worker_id=row["worker_id"],
        lease_acquired_at=row["lease_acquired_at"],
        lease_expires_at=row["lease_expires_at"],
        heartbeat_at=row["heartbeat_at"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        result=result,
        error_code=row["error_code"],
    )


def _claimed_from_row(row: Any) -> ClaimedJob:
    record = _record_from_row(row)
    return ClaimedJob(**record.model_dump(), payload=json.loads(str(row["payload_json"])))


def _safe_json(value: dict[str, Any], maximum: int, label: str) -> str:
    _reject_forbidden_keys(value)
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}_not_json") from exc
    if len(encoded.encode("utf-8")) > maximum:
        raise ValueError(f"{label}_too_large")
    return encoded


def _reject_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() in _FORBIDDEN_STRUCTURED_KEYS:
                raise ValueError("unsafe_structured_field")
            _reject_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_keys(child)


def _bounded_identifier(value: str) -> str:
    if not value or len(value) > 128:
        raise ValueError("invalid_identifier")
    return value


def _optional_bounded_identifier(value: str | None) -> str | None:
    return None if value is None else _bounded_identifier(value)


def _approval_scope_matches(
    *,
    approval_args: dict[str, Any],
    metadata: dict[str, Any],
    conversation_id: str | None,
    project_id: str | None,
) -> bool:
    requested = {
        "conversation_id": conversation_id,
        "project_id": project_id,
    }
    for field, requested_value in requested.items():
        for source in (approval_args, metadata):
            if field in source and source[field] != requested_value:
                return False
    return True


def _bounded_code(value: str, label: str) -> str:
    if (
        not value
        or len(value) > MAX_JOB_EVENT_CODE_CHARS
        or any(char in value for char in "\r\n\x00")
    ):
        raise ValueError(f"invalid_{label}")
    return value


def _progress(value: int) -> int:
    if type(value) is not int or not 0 <= value <= 100:
        raise ValueError("invalid_progress")
    return value
