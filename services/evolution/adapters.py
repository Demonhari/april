from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from april_common.adapter_pointer import (
    active_adapter_path_from_pointer,  # noqa: F401 - compatibility re-export
    adapter_pointer_path,
    read_adapter_pointer,
    sha256_file,
)
from april_common.audit import AuditLogger
from april_common.errors import AprilError, PermissionDeniedError
from april_common.path_security import PathPolicy, normalize_existing_path
from april_common.report_freshness import freshness_from_payload
from april_common.settings import AprilSettings
from april_common.time import utc_now_iso
from services.evolution.write_guard import EvolutionWriteGuard
from services.memory.database import Database

AdapterActionStatus = Literal["activated", "rolled_back", "blocked"]

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def inspect_adapter_state(settings: AprilSettings) -> dict[str, object]:
    """Read-only adapter consistency probe for offline readiness and doctor."""

    database_path = settings.database_path.expanduser().resolve(strict=False)
    if not database_path.is_file():
        return {
            "status": "not_initialized",
            "consistent": True,
            "incomplete_operation_count": 0,
            "database_pointer_disagreement_count": 0,
            "invalid_pointer_count": 0,
        }
    connection = sqlite3.connect(f"{database_path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        tables = {
            str(row["name"])
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table'
                  AND name IN ('adapter_operations', 'model_adapters')
                """
            )
        }
        if "model_adapters" not in tables:
            return {
                "status": "not_initialized",
                "consistent": True,
                "incomplete_operation_count": 0,
                "database_pointer_disagreement_count": 0,
                "invalid_pointer_count": 0,
            }
        pending = (
            list(
                connection.execute(
                    """
                    SELECT model_id FROM adapter_operations
                    WHERE status NOT IN ('completed', 'compensated')
                    """
                )
            )
            if "adapter_operations" in tables
            else []
        )
        active_rows = list(
            connection.execute(
                """
                SELECT model_id, id FROM model_adapters
                WHERE status = 'active'
                ORDER BY model_id, id
                """
            )
        )
    finally:
        connection.close()

    active_by_model: dict[str, list[str]] = {}
    for row in active_rows:
        active_by_model.setdefault(str(row["model_id"]), []).append(str(row["id"]))
    model_ids = set(active_by_model)
    model_ids.update(str(row["model_id"]) for row in pending)
    pointers_dir = settings.evolution_path / "adapters"
    if pointers_dir.is_dir():
        model_ids.update(path.stem for path in pointers_dir.glob("*.json"))
    disagreement_count = 0
    invalid_pointer_count = 0
    for model_id in sorted(model_ids):
        try:
            pointer_version = _effective_pointer_version(
                read_adapter_pointer(settings.home, model_id)
            )
        except (OSError, ValueError, json.JSONDecodeError):
            invalid_pointer_count += 1
            continue
        active_ids = active_by_model.get(model_id, [])
        database_version = (
            _adapter_id_version(active_ids[0], model_id) if len(active_ids) == 1 else None
        )
        if len(active_ids) > 1 or database_version != pointer_version:
            disagreement_count += 1
    consistent = not pending and disagreement_count == 0 and invalid_pointer_count == 0
    return {
        "status": "ok" if consistent else "degraded",
        "consistent": consistent,
        "incomplete_operation_count": len(pending),
        "database_pointer_disagreement_count": disagreement_count,
        "invalid_pointer_count": invalid_pointer_count,
    }


@dataclass(frozen=True, slots=True)
class AdapterActionResult:
    status: AdapterActionStatus
    model_id: str
    version: int | None = None
    active_version: int | None = None
    previous_version: int | None = None
    adapter_path: str | None = None
    sha256: str | None = None
    reason: str | None = None
    next_command: str | None = None

    def to_payload(self) -> dict[str, object | None]:
        return asdict(self)


class AdapterLifecycleManager:
    """Versioned LoRA adapter pointer manager.

    The API side owns DB history and fenced pointer writes. The Runtime side only
    reads the pointer file via :func:`active_adapter_path_from_pointer`, so it
    never opens the Core API SQLite database.
    """

    def __init__(
        self,
        settings: AprilSettings,
        database: Database,
        *,
        audit: AuditLogger | None = None,
        guard: EvolutionWriteGuard | None = None,
    ) -> None:
        self.settings = settings
        self.database = database
        self.audit = audit
        self.guard = guard or EvolutionWriteGuard(settings, audit=audit)

    async def list(self, *, model_id: str | None = None) -> list[dict[str, Any]]:
        if model_id is not None:
            pointer = read_adapter_pointer(self.settings.home, model_id)
            rows = await self.database.fetchall(
                "SELECT * FROM model_adapters WHERE model_id = ? ORDER BY created_at",
                (model_id,),
            )
            return [_adapter_listing(model_id, pointer, rows)]
        model_ids: set[str] = set()
        pointers_dir = self.settings.evolution_path / "adapters"
        if pointers_dir.is_dir():
            model_ids.update(path.stem for path in pointers_dir.glob("*.json"))
        rows = await self.database.fetchall(
            "SELECT DISTINCT model_id FROM model_adapters ORDER BY model_id"
        )
        model_ids.update(str(row["model_id"]) for row in rows)
        return [
            _adapter_listing(
                item,
                read_adapter_pointer(self.settings.home, item),
                await self.database.fetchall(
                    "SELECT * FROM model_adapters WHERE model_id = ? ORDER BY created_at",
                    (item,),
                ),
            )
            for item in sorted(model_ids)
        ]

    async def activate(
        self,
        *,
        model_id: str,
        adapter_path: Path,
        evidence_path: Path | None,
        verification_report_path: Path | None = None,
    ) -> AdapterActionResult:
        _validate_model_id(model_id)
        if self.settings.environment == "production":
            return AdapterActionResult(
                status="blocked",
                model_id=model_id,
                reason=(
                    "direct production adapter activation is disabled; create a "
                    "Phase 4B rollout. Use an isolated candidate Runtime instance "
                    "and exact canary/activation approvals."
                ),
                next_command=(
                    "run april evolve rollout create --type lora_adapter "
                    f"--target-id {model_id} ..."
                ),
            )
        try:
            resolved_adapter = self._normalize_existing_path(adapter_path)
        except (AprilError, OSError) as exc:
            return AdapterActionResult(
                status="blocked",
                model_id=model_id,
                reason=f"adapter path is not readable: {exc}",
            )
        adapter_sha = sha256_file(resolved_adapter)
        gate = self._validate_perplexity_evidence(
            model_id=model_id,
            adapter_path=resolved_adapter,
            adapter_sha256=adapter_sha,
            evidence_path=evidence_path,
        )
        if gate is not None:
            return gate
        if self.settings.environment == "production":
            report_gate = self._validate_real_model_report(
                model_id=model_id,
                adapter_path=resolved_adapter,
                adapter_sha256=adapter_sha,
                report_path=verification_report_path,
            )
            if report_gate is not None:
                return report_gate

        async with self.database.write_coordination():
            await self._reconcile_incomplete_locked(model_id=model_id)
            previous_pointer = read_adapter_pointer(self.settings.home, model_id)
            target_pointer = _copy_pointer(previous_pointer or _empty_pointer(model_id))
            previous = _pointer_active_version(target_pointer)
            version = _next_pointer_version(target_pointer)
            created_at = utc_now_iso()
            target_pointer["versions"].append(
                {
                    "version": version,
                    "adapter_path": str(resolved_adapter),
                    "sha256": adapter_sha,
                    "created_at": created_at,
                }
            )
            target_pointer["active_version"] = version
            target_pointer["sha256"] = adapter_sha
            target_pointer["updated_at"] = created_at
            if target_pointer.get("created_at") is None:
                target_pointer["created_at"] = created_at
            await self._run_operation_locked(
                operation_id=str(uuid.uuid4()),
                operation_type="activate",
                model_id=model_id,
                previous_pointer=previous_pointer,
                target_pointer=target_pointer,
                target_path=resolved_adapter,
                target_sha256=adapter_sha,
                eval_score=_adapter_ppl_from_evidence(evidence_path),
                baseline_score=_base_ppl_from_evidence(evidence_path),
                created_at=created_at,
            )
            self._audit(
                "adapter_activated",
                model_id=model_id,
                version=version,
                previous_version=previous,
                sha256=adapter_sha,
            )
            return AdapterActionResult(
                status="activated",
                model_id=model_id,
                version=version,
                active_version=version,
                previous_version=previous,
                adapter_path=str(resolved_adapter),
                sha256=adapter_sha,
            )

    async def rollback(
        self,
        *,
        model_id: str,
        version: int | None = None,
    ) -> AdapterActionResult:
        _validate_model_id(model_id)
        async with self.database.write_coordination():
            await self._reconcile_incomplete_locked(model_id=model_id)
            pointer = read_adapter_pointer(self.settings.home, model_id)
            if pointer is None:
                return AdapterActionResult(
                    status="blocked",
                    model_id=model_id,
                    reason="no adapter pointer exists for model",
                )
            previous = _pointer_active_version(pointer)
            target = version if version is not None else _previous_pointer_version(pointer)
            if target is None:
                return AdapterActionResult(
                    status="blocked",
                    model_id=model_id,
                    active_version=previous,
                    reason="no previous adapter version is available",
                )
            target_entry = _pointer_entry(pointer, target)
            if target_entry is None:
                return AdapterActionResult(
                    status="blocked",
                    model_id=model_id,
                    active_version=previous,
                    version=target,
                    reason="target adapter version not found",
                )
            try:
                target_path = self._normalize_existing_path(Path(str(target_entry["adapter_path"])))
            except (AprilError, OSError) as exc:
                return AdapterActionResult(
                    status="blocked",
                    model_id=model_id,
                    active_version=previous,
                    version=target,
                    reason=f"target adapter path is not readable: {exc}",
                )
            target_sha = str(target_entry["sha256"])
            if not _valid_adapter_file(target_path, target_sha):
                return AdapterActionResult(
                    status="blocked",
                    model_id=model_id,
                    active_version=previous,
                    version=target,
                    reason=f"target adapter file is missing or changed: {target_path}",
                )
            target_pointer = _copy_pointer(pointer)
            target_pointer["active_version"] = target
            target_pointer["sha256"] = target_sha
            target_pointer["updated_at"] = utc_now_iso()
            await self._run_operation_locked(
                operation_id=str(uuid.uuid4()),
                operation_type="rollback",
                model_id=model_id,
                previous_pointer=pointer,
                target_pointer=target_pointer,
                target_path=target_path,
                target_sha256=target_sha,
                eval_score=None,
                baseline_score=None,
                created_at=utc_now_iso(),
            )
            self._audit(
                "adapter_rolled_back",
                model_id=model_id,
                version=target,
                previous_version=previous,
                sha256=target_sha,
            )
            return AdapterActionResult(
                status="rolled_back",
                model_id=model_id,
                version=target,
                active_version=target,
                previous_version=previous,
                adapter_path=str(target_path),
                sha256=target_sha,
            )

    async def reconcile_incomplete_operations(self) -> dict[str, object]:
        """Converge interrupted adapter operations before services become ready."""

        async with self.database.write_coordination():
            recovered = await self._reconcile_incomplete_locked()
        health = await self.state_health()
        return {**health, "recovered_operation_count": recovered}

    async def state_health(self) -> dict[str, object]:
        pending_rows = await self.database.fetchall(
            """
            SELECT id, model_id, status
            FROM adapter_operations
            WHERE status NOT IN ('completed', 'compensated')
            ORDER BY created_at, id
            """
        )
        model_rows = await self.database.fetchall(
            """
            SELECT model_id, id
            FROM model_adapters
            WHERE status = 'active'
            ORDER BY model_id, id
            """
        )
        active_by_model: dict[str, list[str]] = {}
        for row in model_rows:
            active_by_model.setdefault(str(row["model_id"]), []).append(str(row["id"]))
        model_ids = set(active_by_model)
        model_ids.update(str(row["model_id"]) for row in pending_rows)
        pointers_dir = self.settings.evolution_path / "adapters"
        if pointers_dir.is_dir():
            model_ids.update(path.stem for path in pointers_dir.glob("*.json"))

        disagreement_count = 0
        invalid_pointer_count = 0
        for model_id in sorted(model_ids):
            try:
                pointer = read_adapter_pointer(self.settings.home, model_id)
                pointer_version = _effective_pointer_version(pointer)
            except (OSError, ValueError, json.JSONDecodeError):
                invalid_pointer_count += 1
                continue
            active_ids = active_by_model.get(model_id, [])
            database_version = (
                _adapter_id_version(active_ids[0], model_id) if len(active_ids) == 1 else None
            )
            if len(active_ids) > 1 or database_version != pointer_version:
                disagreement_count += 1
        consistent = not pending_rows and disagreement_count == 0 and invalid_pointer_count == 0
        return {
            "status": "ok" if consistent else "degraded",
            "consistent": consistent,
            "incomplete_operation_count": len(pending_rows),
            "database_pointer_disagreement_count": disagreement_count,
            "invalid_pointer_count": invalid_pointer_count,
        }

    async def _run_operation_locked(
        self,
        *,
        operation_id: str,
        operation_type: Literal["activate", "rollback"],
        model_id: str,
        previous_pointer: dict[str, Any] | None,
        target_pointer: dict[str, Any],
        target_path: Path,
        target_sha256: str,
        eval_score: float | None,
        baseline_score: float | None,
        created_at: str,
    ) -> None:
        previous_version = _effective_pointer_version(previous_pointer)
        target_version = _pointer_active_version(target_pointer)
        if target_version is None:
            raise ValueError("Adapter operation target must have an active version.")
        self.guard.validate_table("adapter_operations")
        self.guard.validate_table("model_adapters")
        try:
            async with self.database.transaction_under_coordination() as conn:
                await conn.execute(
                    """
                    INSERT INTO adapter_operations(
                        id, model_id, operation_type, status,
                        previous_active_version, requested_target_version,
                        previous_pointer_json, target_pointer_json,
                        target_adapter_path, target_sha256,
                        created_at, updated_at
                    )
                    VALUES(?, ?, ?, 'prepared', ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        operation_id,
                        model_id,
                        operation_type,
                        previous_version,
                        target_version,
                        _pointer_json(previous_pointer),
                        _pointer_json(target_pointer),
                        str(target_path),
                        target_sha256,
                        created_at,
                        created_at,
                    ),
                )

            pending_pointer = _copy_pointer(target_pointer)
            pending_pointer["pending_operation"] = {
                "id": operation_id,
                "previous_active_version": previous_version,
            }
            self._write_pointer(model_id, pending_pointer)
            await self._set_operation_status_locked(operation_id, "pointer_switched")
            await self._commit_target_state_locked(
                operation_id=operation_id,
                operation_type=operation_type,
                model_id=model_id,
                target_version=target_version,
                target_path=target_path,
                eval_score=eval_score,
                baseline_score=baseline_score,
                created_at=created_at,
            )
            self._write_pointer(model_id, target_pointer)
            await self._set_operation_status_locked(
                operation_id,
                "completed",
                completed=True,
            )
        except BaseException:
            recovery = asyncio.create_task(
                self._recover_operation_locked(operation_id, failure_path=True)
            )
            try:
                await asyncio.shield(recovery)
            except BaseException:
                if not recovery.done():
                    await asyncio.shield(recovery)
            raise

    async def _commit_target_state_locked(
        self,
        *,
        operation_id: str,
        operation_type: str,
        model_id: str,
        target_version: int,
        target_path: Path,
        eval_score: float | None,
        baseline_score: float | None,
        created_at: str,
    ) -> None:
        async with self.database.transaction_under_coordination() as conn:
            await conn.execute(
                "UPDATE model_adapters SET status = 'inactive' WHERE model_id = ?",
                (model_id,),
            )
            if operation_type == "activate":
                await conn.execute(
                    """
                    INSERT INTO model_adapters(
                        id, model_id, adapter_path, status, eval_score,
                        baseline_score, created_at, activated_at
                    )
                    VALUES(?, ?, ?, 'active', ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        adapter_path = excluded.adapter_path,
                        status = 'active',
                        eval_score = excluded.eval_score,
                        baseline_score = excluded.baseline_score,
                        activated_at = excluded.activated_at
                    """,
                    (
                        f"{model_id}:{target_version}",
                        model_id,
                        str(target_path),
                        eval_score,
                        baseline_score,
                        created_at,
                        created_at,
                    ),
                )
            else:
                cursor = await conn.execute(
                    """
                    UPDATE model_adapters
                    SET status = 'active', activated_at = ?
                    WHERE id = ? AND model_id = ?
                    """,
                    (created_at, f"{model_id}:{target_version}", model_id),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("Rollback target is absent from adapter history.")
            await conn.execute(
                """
                UPDATE adapter_operations
                SET status = 'database_committed', updated_at = ?, error_code = NULL
                WHERE id = ?
                """,
                (utc_now_iso(), operation_id),
            )

    async def _set_operation_status_locked(
        self,
        operation_id: str,
        status: str,
        *,
        completed: bool = False,
        error_code: str | None = None,
    ) -> None:
        async with self.database.transaction_under_coordination() as conn:
            await conn.execute(
                """
                UPDATE adapter_operations
                SET status = ?, updated_at = ?, completed_at = ?, error_code = ?
                WHERE id = ?
                """,
                (
                    status,
                    utc_now_iso(),
                    utc_now_iso() if completed else None,
                    error_code,
                    operation_id,
                ),
            )

    async def _reconcile_incomplete_locked(self, *, model_id: str | None = None) -> int:
        where = "status NOT IN ('completed', 'compensated')"
        parameters: tuple[Any, ...] = ()
        if model_id is not None:
            where += " AND model_id = ?"
            parameters = (model_id,)
        rows = await self.database.fetchall(
            f"""
            SELECT *
            FROM adapter_operations
            WHERE {where}
            ORDER BY created_at, id
            """,
            parameters,
        )
        recovered = 0
        for row in rows:
            await self._recover_operation_locked(str(row["id"]), failure_path=False)
            recovered += 1
        return recovered

    async def _recover_operation_locked(
        self,
        operation_id: str,
        *,
        failure_path: bool,
    ) -> None:
        row = await self.database.fetchone(
            "SELECT * FROM adapter_operations WHERE id = ?",
            (operation_id,),
        )
        if row is None or str(row["status"]) in {"completed", "compensated"}:
            return
        target_pointer = _load_pointer_json(str(row["target_pointer_json"]))
        previous_pointer = _load_pointer_json(row["previous_pointer_json"])
        model_id = str(row["model_id"])
        status = str(row["status"])
        target_path = Path(str(row["target_adapter_path"])).expanduser().resolve(strict=False)
        target_sha = str(row["target_sha256"])

        if (
            target_pointer is not None
            and status == "database_committed"
            and _valid_adapter_file(target_path, target_sha)
        ):
            try:
                self._write_pointer(model_id, target_pointer)
                await self._set_operation_status_locked(
                    operation_id,
                    "completed",
                    completed=True,
                )
                self._audit_recovery(
                    model_id=model_id,
                    operation_type=str(row["operation_type"]),
                    outcome="completed",
                    target_version=int(row["requested_target_version"]),
                    failure_path=failure_path,
                )
                return
            except BaseException:
                pass

        await self._compensate_operation_locked(
            operation_id=operation_id,
            model_id=model_id,
            previous_pointer=previous_pointer,
            previous_version=(
                int(row["previous_active_version"])
                if row["previous_active_version"] is not None
                else None
            ),
        )
        self._audit_recovery(
            model_id=model_id,
            operation_type=str(row["operation_type"]),
            outcome="compensated",
            target_version=int(row["requested_target_version"]),
            failure_path=failure_path,
        )

    async def _compensate_operation_locked(
        self,
        *,
        operation_id: str,
        model_id: str,
        previous_pointer: dict[str, Any] | None,
        previous_version: int | None,
    ) -> None:
        safe_pointer = _safe_previous_pointer(
            previous_pointer,
            root=self.settings.home,
            allowed_roots=tuple(self.settings.allowed_roots),
        )
        safe_version = (
            previous_version
            if safe_pointer is not None
            and _effective_pointer_version(safe_pointer) == previous_version
            else None
        )
        async with self.database.transaction_under_coordination() as conn:
            await conn.execute(
                "UPDATE model_adapters SET status = 'inactive' WHERE model_id = ?",
                (model_id,),
            )
            if safe_version is not None:
                cursor = await conn.execute(
                    """
                    UPDATE model_adapters
                    SET status = 'active', activated_at = ?
                    WHERE id = ? AND model_id = ?
                    """,
                    (utc_now_iso(), f"{model_id}:{safe_version}", model_id),
                )
                if cursor.rowcount != 1:
                    safe_pointer = None
                    safe_version = None
            await conn.execute(
                """
                UPDATE adapter_operations
                SET status = 'compensation_pending', updated_at = ?,
                    completed_at = NULL, error_code = 'operation_interrupted'
                WHERE id = ?
                """,
                (utc_now_iso(), operation_id),
            )
        self._restore_pointer(model_id, safe_pointer)
        await self._set_operation_status_locked(
            operation_id,
            "compensated",
            completed=True,
            error_code="operation_interrupted",
        )

    def _restore_pointer(self, model_id: str, pointer: dict[str, Any] | None) -> None:
        if pointer is not None:
            restored = _copy_pointer(pointer)
            restored.pop("pending_operation", None)
            self._write_pointer(model_id, restored)
            return
        self.guard.remove_file(adapter_pointer_path(self.settings.home, model_id))

    def _audit_recovery(
        self,
        *,
        model_id: str,
        operation_type: str,
        outcome: str,
        target_version: int,
        failure_path: bool,
    ) -> None:
        if self.audit is not None:
            self.audit.write(
                {
                    "event_type": "adapter_operation_recovery",
                    "actor": "april-core",
                    "model_id": model_id,
                    "operation_type": operation_type,
                    "outcome": outcome,
                    "target_version": target_version,
                    "failure_path": failure_path,
                }
            )

    def _validate_perplexity_evidence(
        self,
        *,
        model_id: str,
        adapter_path: Path,
        adapter_sha256: str,
        evidence_path: Path | None,
    ) -> AdapterActionResult | None:
        if evidence_path is None:
            return AdapterActionResult(
                status="blocked",
                model_id=model_id,
                reason="missing perplexity evidence JSON",
                next_command=_perplexity_next_command(model_id, adapter_path),
            )
        try:
            normalized_evidence_path = self._normalize_existing_text_path(evidence_path)
            evidence = json.loads(normalized_evidence_path.read_text(encoding="utf-8"))
        except (AprilError, OSError, json.JSONDecodeError) as exc:
            return AdapterActionResult(
                status="blocked",
                model_id=model_id,
                reason=f"invalid perplexity evidence JSON: {exc}",
                next_command=_perplexity_next_command(model_id, adapter_path),
            )
        if not isinstance(evidence, dict):
            return AdapterActionResult(
                status="blocked",
                model_id=model_id,
                reason="perplexity evidence must be a JSON object",
                next_command=_perplexity_next_command(model_id, adapter_path),
            )
        if str(evidence.get("evidence_type")) != "lora_perplexity":
            return AdapterActionResult(
                status="blocked",
                model_id=model_id,
                reason="perplexity evidence_type must be lora_perplexity",
                next_command=_perplexity_next_command(model_id, adapter_path),
            )
        if str(evidence.get("model_id")) != model_id:
            return AdapterActionResult(
                status="blocked",
                model_id=model_id,
                reason="perplexity evidence model_id does not match activation model_id",
            )
        evidence_sha = str(evidence.get("adapter_sha256") or "")
        if not _SHA256_RE.fullmatch(evidence_sha) or evidence_sha != adapter_sha256:
            return AdapterActionResult(
                status="blocked",
                model_id=model_id,
                reason="perplexity evidence adapter_sha256 does not match adapter file",
            )
        base = _evidence_float(evidence, "base_perplexity", "base_ppl")
        adapter = _evidence_float(evidence, "adapter_perplexity", "adapter_ppl")
        if base is None or adapter is None:
            return AdapterActionResult(
                status="blocked",
                model_id=model_id,
                reason="perplexity evidence must include base and adapter perplexity",
            )
        if adapter > base:
            return AdapterActionResult(
                status="blocked",
                model_id=model_id,
                reason=(
                    f"adapter perplexity {adapter:.4g} is worse than base perplexity {base:.4g}"
                ),
            )
        return None

    def _validate_real_model_report(
        self,
        *,
        model_id: str,
        adapter_path: Path,
        adapter_sha256: str,
        report_path: Path | None,
    ) -> AdapterActionResult | None:
        if report_path is None:
            return AdapterActionResult(
                status="blocked",
                model_id=model_id,
                reason="production activation requires a fresh real-model verification report",
                next_command=_verification_next_command(model_id, adapter_path),
            )
        try:
            normalized_report_path = self._normalize_existing_text_path(report_path)
            payload = json.loads(normalized_report_path.read_text(encoding="utf-8"))
        except (AprilError, OSError, json.JSONDecodeError) as exc:
            return AdapterActionResult(
                status="blocked",
                model_id=model_id,
                reason=f"invalid real-model verification report JSON: {exc}",
                next_command=_verification_next_command(model_id, adapter_path),
            )
        if not isinstance(payload, dict):
            return AdapterActionResult(
                status="blocked",
                model_id=model_id,
                reason="real-model verification report must be a JSON object",
                next_command=_verification_next_command(model_id, adapter_path),
            )
        from april_common.config_fingerprint import config_fingerprint_digest

        if (
            payload.get("report_type") != "multi_model"
            or payload.get("runtime_backend") != "llama_cpp"
        ):
            return AdapterActionResult(
                status="blocked",
                model_id=model_id,
                reason="verification report must be a llama_cpp multi_model report",
                next_command=_verification_next_command(model_id, adapter_path),
            )
        freshness = freshness_from_payload(
            payload,
            report_type="multi_model",
            current_fingerprint=config_fingerprint_digest(self.settings.home),
            basename=report_path.name,
        )
        if freshness.stale:
            return AdapterActionResult(
                status="blocked",
                model_id=model_id,
                reason=f"verification report is stale: {freshness.stale_reason}",
                next_command=_verification_next_command(model_id, adapter_path),
            )
        for item in payload.get("models", []):
            if not isinstance(item, dict) or item.get("model_id") != model_id:
                continue
            if item.get("load_success") is not True:
                break
            if item.get("adapter_sha256") == adapter_sha256:
                return None
            break
        return AdapterActionResult(
            status="blocked",
            model_id=model_id,
            reason="verification report did not load this adapter for the model",
            next_command=_verification_next_command(model_id, adapter_path),
        )

    def _write_pointer(self, model_id: str, pointer: dict[str, Any]) -> Path:
        return self.guard.write_text(
            adapter_pointer_path(self.settings.home, model_id),
            json.dumps(pointer, indent=2, sort_keys=True) + "\n",
        )

    def _normalize_existing_path(self, path: Path) -> Path:
        policy = PathPolicy(
            allowed_roots=tuple(self.settings.allowed_roots),
            max_read_bytes=self.settings.paths.max_file_read_bytes,
            max_write_bytes=self.settings.paths.max_file_write_bytes,
        )
        return normalize_existing_path(path, policy)

    def _normalize_existing_text_path(self, path: Path) -> Path:
        normalized = self._normalize_existing_path(path)
        if normalized.stat().st_size > self.settings.paths.max_file_read_bytes:
            raise PermissionDeniedError(
                "Evidence/report file exceeds configured maximum read size."
            )
        return normalized

    def _audit(
        self,
        event_type: str,
        *,
        model_id: str,
        version: int,
        previous_version: int | None,
        sha256: str,
    ) -> None:
        if self.audit is not None:
            self.audit.write(
                {
                    "event_type": event_type,
                    "actor": "local-user",
                    "model_id": model_id,
                    "version": version,
                    "previous_version": previous_version,
                    "sha256": sha256,
                }
            )


def _empty_pointer(model_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_id": model_id,
        "created_at": None,
        "updated_at": None,
        "active_version": None,
        "sha256": None,
        "versions": [],
    }


def _copy_pointer(pointer: dict[str, Any]) -> dict[str, Any]:
    return json.loads(_pointer_json(pointer) or "{}")


def _pointer_json(pointer: dict[str, Any] | None) -> str | None:
    if pointer is None:
        return None
    return json.dumps(
        pointer,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _load_pointer_json(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    payload = json.loads(str(value))
    if not isinstance(payload, dict):
        raise ValueError("Adapter operation pointer snapshot must be an object.")
    return payload


def _effective_pointer_version(pointer: dict[str, Any] | None) -> int | None:
    if pointer is None:
        return None
    pending = pointer.get("pending_operation")
    if isinstance(pending, dict):
        value = pending.get("previous_active_version")
        return int(value) if value is not None else None
    return _pointer_active_version(pointer)


def _valid_adapter_file(path: Path, expected_sha256: str) -> bool:
    if not _SHA256_RE.fullmatch(expected_sha256) or not path.is_file():
        return False
    try:
        return sha256_file(path) == expected_sha256
    except OSError:
        return False


def _safe_previous_pointer(
    pointer: dict[str, Any] | None,
    *,
    root: Path,
    allowed_roots: tuple[Path, ...],
) -> dict[str, Any] | None:
    if pointer is None:
        return None
    restored = _copy_pointer(pointer)
    restored.pop("pending_operation", None)
    active = _pointer_active_version(restored)
    if active is None:
        return restored
    entry = _pointer_entry(restored, active)
    if entry is None:
        return None
    path = Path(str(entry.get("adapter_path") or "")).expanduser()
    path = path.resolve(strict=False) if path.is_absolute() else (root / path).resolve(strict=False)
    within_allowed_root = any(
        _is_relative_to(path, allowed_root.resolve(strict=False)) for allowed_root in allowed_roots
    )
    if not within_allowed_root:
        return None
    return restored if _valid_adapter_file(path, str(entry.get("sha256") or "")) else None


def _adapter_id_version(adapter_id: str, model_id: str) -> int | None:
    prefix = f"{model_id}:"
    if not adapter_id.startswith(prefix):
        return None
    try:
        return int(adapter_id[len(prefix) :])
    except ValueError:
        return None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _adapter_listing(
    model_id: str,
    pointer: dict[str, Any] | None,
    rows: list[Any],
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "pointer": pointer,
        "history": [dict(row) for row in rows],
    }


def _next_pointer_version(pointer: dict[str, Any]) -> int:
    versions = pointer.get("versions")
    if not isinstance(versions, list):
        return 1
    seen = [int(item.get("version", 0)) for item in versions if isinstance(item, dict)]
    return max(seen, default=0) + 1


def _previous_pointer_version(pointer: dict[str, Any]) -> int | None:
    active = _pointer_active_version(pointer)
    versions = pointer.get("versions")
    if active is None or not isinstance(versions, list):
        return None
    ordered = [int(item["version"]) for item in versions if isinstance(item, dict)]
    prior = [version for version in ordered if version < active]
    return prior[-1] if prior else None


def _pointer_active_version(pointer: dict[str, Any]) -> int | None:
    value = pointer.get("active_version")
    if value is None:
        return None
    return int(value)


def _pointer_entry(pointer: dict[str, Any], version: int) -> dict[str, Any] | None:
    versions = pointer.get("versions")
    if not isinstance(versions, list):
        return None
    for item in versions:
        if isinstance(item, dict) and int(item.get("version", 0)) == version:
            return item
    return None


def _validate_model_id(model_id: str) -> None:
    if not _MODEL_ID_RE.fullmatch(model_id):
        raise ValueError("model_id is not safe for an adapter pointer filename")


def _evidence_float(evidence: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = evidence.get(key)
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _read_evidence(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _adapter_ppl_from_evidence(path: Path | None) -> float | None:
    evidence = _read_evidence(path)
    return _evidence_float(evidence or {}, "adapter_perplexity", "adapter_ppl")


def _base_ppl_from_evidence(path: Path | None) -> float | None:
    evidence = _read_evidence(path)
    return _evidence_float(evidence or {}, "base_perplexity", "base_ppl")


def _perplexity_next_command(model_id: str, adapter_path: Path) -> str:
    return (
        ".venv/bin/python scripts/finetune/write_perplexity_evidence.py "
        f"--model-id {model_id} --adapter-path {adapter_path} "
        "--base-ppl <BASE_PPL> --adapter-ppl <ADAPTER_PPL> "
        "--heldout-dataset <HELDOUT_JSONL> "
        f"--output data/evolution/adapters/evidence/{model_id}.perplexity.json"
    )


def _verification_next_command(model_id: str, adapter_path: Path) -> str:
    return (
        "APRIL_RUNTIME_BACKEND=llama_cpp .venv/bin/python -m apps.runner.main "
        "april verify --all-configured-models --require-real-model "
        f"--candidate-adapter-model-id {model_id} --candidate-adapter-path {adapter_path} "
        f"--report data/verification/adapter-{model_id}.json"
    )
