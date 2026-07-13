from __future__ import annotations

import json
import re
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

        pointer = read_adapter_pointer(self.settings.home, model_id) or _empty_pointer(model_id)
        previous = _pointer_active_version(pointer)
        version = _next_pointer_version(pointer)
        created_at = utc_now_iso()
        entry = {
            "version": version,
            "adapter_path": str(resolved_adapter),
            "sha256": adapter_sha,
            "created_at": created_at,
        }
        pointer["versions"].append(entry)
        pointer["active_version"] = version
        pointer["sha256"] = adapter_sha
        pointer["updated_at"] = created_at
        if pointer.get("created_at") is None:
            pointer["created_at"] = created_at
        self._write_pointer(model_id, pointer)

        self.guard.validate_table("model_adapters")
        async with self.database.transaction() as conn:
            await conn.execute(
                "UPDATE model_adapters SET status = 'inactive' WHERE model_id = ?",
                (model_id,),
            )
            await conn.execute(
                """
                INSERT INTO model_adapters(
                    id, model_id, adapter_path, status, eval_score,
                    baseline_score, created_at, activated_at
                )
                VALUES(?, ?, ?, 'active', ?, ?, ?, ?)
                """,
                (
                    f"{model_id}:{version}",
                    model_id,
                    str(resolved_adapter),
                    _adapter_ppl_from_evidence(evidence_path),
                    _base_ppl_from_evidence(evidence_path),
                    created_at,
                    created_at,
                ),
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
                reason="target adapter version not found",
            )
        target_path = Path(str(target_entry["adapter_path"])).expanduser().resolve(strict=False)
        if not target_path.exists():
            return AdapterActionResult(
                status="blocked",
                model_id=model_id,
                active_version=previous,
                version=target,
                reason=f"target adapter file is missing: {target_path}",
            )
        pointer["active_version"] = target
        pointer["sha256"] = str(target_entry["sha256"])
        pointer["updated_at"] = utc_now_iso()
        self._write_pointer(model_id, pointer)
        self.guard.validate_table("model_adapters")
        async with self.database.transaction() as conn:
            await conn.execute(
                "UPDATE model_adapters SET status = 'inactive' WHERE model_id = ?",
                (model_id,),
            )
            await conn.execute(
                "UPDATE model_adapters SET status = 'active', activated_at = ? WHERE id = ?",
                (utc_now_iso(), f"{model_id}:{target}"),
            )
        self._audit(
            "adapter_rolled_back",
            model_id=model_id,
            version=target,
            previous_version=previous,
            sha256=str(target_entry["sha256"]),
        )
        return AdapterActionResult(
            status="rolled_back",
            model_id=model_id,
            version=target,
            active_version=target,
            previous_version=previous,
            adapter_path=str(target_path),
            sha256=str(target_entry["sha256"]),
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
