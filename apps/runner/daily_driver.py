"""``run april doctor --daily-driver`` — the normal pre-use readiness summary.

This is the one command to run before relying on APRIL day to day. It folds the
already-redacted, offline primitives (config validation, offline readiness,
report freshness, the config fingerprint, the memory/embedding doctor signals,
and the voice milestone) into a single verdict with three headline rollups —
**core real model**, **workflow security**, and **hardened go-live** — plus a
per-check table and the exact next commands.

It is inert and safe by construction: it only reads ``configs``/settings, local
report JSON, and ``Path.exists`` probes. It never opens the microphone, never
loads a model (heavy real verification is opt-in via ``--run-real-checks`` on the
CLI, which runs the existing verifiers *before* this read-only summary), never
mutates configuration, and never prints tokens, prompts, transcripts, patch
bytes, or absolute private paths — every emitted field is a status enum, a count,
a basename, a redacted reason, or a copy-pasteable command.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from apps.runner.mac_report import redact_reason
from apps.runner.readiness import ReadinessReport, build_readiness_report
from apps.runner.reports import classify_report, load_reports
from april_common.config_fingerprint import config_fingerprint_digest
from april_common.errors import ConfigError
from april_common.report_freshness import ReportFreshness, freshness_from_payload
from april_common.settings import AprilSettings, load_settings
from april_common.time import utc_now_iso

DailyStatus = Literal["ready", "warning", "blocker", "not_run"]

# Exact, copy-pasteable next commands (never executed here).
_SETUP_TOKENS = "run april setup tokens"
_INSTALL_RUNTIME = "pip install -e '.[runtime]'"
_SETUP_MODELS = "run april setup models"
_CONFIG_VALIDATE = "run april config validate"
_VERIFY_REAL = (
    "run april verify --all-configured-models --require-real-model "
    "--report data/verification/mac-readiness.json"
)
_VERIFY_WORKFLOW = (
    "run april verify --workflow --real-model --report data/verification/workflow-real.json"
)
_GO_LIVE = "run april go-live --write-report --start-services"
_SETUP_EMBEDDINGS = (
    "run april setup embeddings --model /absolute/path/to/embedding.gguf "
    "--id april-embedding --apply"
)
_MEMORY_REINDEX = "run april memory reindex"
_SET_BACKEND = "Set runtime.backend=llama_cpp (or unset APRIL_RUNTIME_BACKEND=fake)."

# severity order for rollups / overall verdict
_SEVERITY = {"ready": 0, "not_run": 1, "warning": 2, "blocker": 3}


class DailyDriverCheck(BaseModel):
    name: str
    status: DailyStatus
    detail: str
    next_command: str | None = None


class DailyDriverReport(BaseModel):
    schema_version: int = 1
    report_type: Literal["daily_driver"] = "daily_driver"
    generated_at: str
    config_fingerprint: str | None = None
    runtime_backend: str = "unknown"
    # Headline rollups.
    core_real_model: DailyStatus = "not_run"
    workflow_security: DailyStatus = "not_run"
    hardened_go_live: DailyStatus = "not_run"
    hardened_reason: str | None = None
    overall: DailyStatus = "not_run"
    checks: list[DailyDriverCheck] = Field(default_factory=list)
    next_commands: list[str] = Field(default_factory=list)


def _worst(statuses: list[DailyStatus]) -> DailyStatus:
    worst: DailyStatus = "ready"
    for status in statuses:
        if _SEVERITY[status] > _SEVERITY[worst]:
            worst = status
    return worst


def _latest_by_type(home: Path) -> dict[str, tuple[Path, dict]]:
    """Newest readable report per classified type under data/verification."""
    latest: dict[str, tuple[Path, dict]] = {}
    for path, payload in load_reports(home / "data" / "verification"):
        report_type = classify_report(payload)
        if report_type == "unknown":
            continue
        # load_reports is already newest-first, so the first seen per type wins.
        latest.setdefault(report_type, (path, payload))
    return latest


def _report_status(payload: dict) -> str | None:
    value = payload.get("final_status") or payload.get("summary")
    return str(value) if value is not None else None


def _path_writable(path: Path) -> bool:
    """Whether ``path`` (or its nearest existing parent) is writable."""
    probe = path
    while not probe.exists():
        if probe.parent == probe:
            return False
        probe = probe.parent
    return os.access(probe, os.W_OK)


def build_daily_driver_report(home: Path) -> DailyDriverReport:
    """Assemble the read-only daily-driver verdict. Pure over local state."""
    home = home.expanduser().resolve()
    fingerprint = config_fingerprint_digest(home)
    checks: list[DailyDriverCheck] = []

    # --- config validation --------------------------------------------------
    config_errors = _config_errors(home)
    config_valid = not config_errors
    checks.append(
        DailyDriverCheck(
            name="config validation",
            status="ready" if config_valid else "blocker",
            detail="configuration is valid"
            if config_valid
            else "; ".join(redact_reason(error) for error in config_errors)[:240],
            next_command=None if config_valid else _CONFIG_VALIDATE,
        )
    )

    readiness = build_readiness_report(home)
    settings = _safe_settings(home)
    latest = _latest_by_type(home)

    # --- backend / runtime --------------------------------------------------
    backend = readiness.runtime_backend
    if readiness.runtime_is_fake:
        checks.append(
            DailyDriverCheck(
                name="runtime backend",
                status="blocker",
                detail=f"backend is '{backend}' (fake/simulated), not llama_cpp",
                next_command=_SET_BACKEND,
            )
        )
    else:
        checks.append(DailyDriverCheck(name="runtime backend", status="ready", detail="llama_cpp"))

    # --- llama-cpp-python ----------------------------------------------------
    if readiness.llama_cpp_python_available:
        checks.append(
            DailyDriverCheck(name="llama-cpp-python", status="ready", detail="import spec found")
        )
    else:
        checks.append(
            DailyDriverCheck(
                name="llama-cpp-python",
                status="blocker",
                detail="optional runtime extra is not installed",
                next_command=_INSTALL_RUNTIME,
            )
        )

    # --- configured GGUF presence -------------------------------------------
    chat_models = [
        m for m in readiness.models if m.backend == "llama_cpp" and m.role != "embedding"
    ]
    missing = [m.id for m in chat_models if not m.path_exists]
    if not chat_models:
        checks.append(
            DailyDriverCheck(
                name="configured GGUF presence",
                status="blocker",
                detail="no llama_cpp chat models are configured",
                next_command=_SETUP_MODELS,
            )
        )
    elif missing:
        checks.append(
            DailyDriverCheck(
                name="configured GGUF presence",
                status="blocker",
                detail="missing model files: " + ", ".join(sorted(missing)),
                next_command=_SETUP_MODELS,
            )
        )
    else:
        checks.append(
            DailyDriverCheck(
                name="configured GGUF presence",
                status="ready",
                detail=f"{len(chat_models)} configured chat GGUF(s) present",
            )
        )

    # --- latest report freshness (real-model / workflow / go-live) ----------
    real_fresh, real_check = _report_check(
        "latest real-model verification", latest.get("multi_model"), fingerprint, _VERIFY_REAL
    )
    checks.append(real_check)
    workflow_fresh, workflow_check = _report_check(
        "latest workflow-real verification", latest.get("workflow"), fingerprint, _VERIFY_WORKFLOW
    )
    checks.append(workflow_check)
    go_live_fresh, go_live_check = _report_check(
        "latest go-live", latest.get("go_live"), fingerprint, _GO_LIVE
    )
    checks.append(go_live_check)

    # --- token hardening -----------------------------------------------------
    checks.append(_token_check(readiness))

    # --- embedding provider --------------------------------------------------
    checks.append(_embedding_check(settings))

    # --- vector index compatibility -----------------------------------------
    checks.append(_vector_index_check(home, settings))

    # --- voice milestone -----------------------------------------------------
    checks.append(_voice_check(readiness, latest))

    # --- desktop readiness (read-only console) ------------------------------
    checks.append(
        DailyDriverCheck(
            name="desktop readiness",
            status="ready",
            detail="read-only operator console available via `run april desktop`",
        )
    )

    # --- report directory ----------------------------------------------------
    reports_dir = home / "data" / "verification"
    checks.append(
        DailyDriverCheck(
            name="report directory",
            status="ready" if _path_writable(reports_dir) else "warning",
            detail="data/verification is writable"
            if _path_writable(reports_dir)
            else "data/verification is not writable",
        )
    )

    # --- audit log -----------------------------------------------------------
    audit_writable = settings is not None and _path_writable(settings.audit_path)
    if settings is None:
        audit_status: DailyStatus = "warning"
        audit_detail = "settings unavailable; audit path unknown"
    elif settings.audit_path.exists():
        audit_status = "ready"
        audit_detail = "audit log present and writable"
    elif audit_writable:
        audit_status = "ready"
        audit_detail = "audit directory writable (log created when services run)"
    else:
        audit_status = "warning"
        audit_detail = "audit directory is not writable"
    checks.append(DailyDriverCheck(name="audit log", status=audit_status, detail=audit_detail))

    # --- rollups -------------------------------------------------------------
    core_real_model = _core_rollup(
        config_valid=config_valid,
        readiness=readiness,
        chat_models_present=bool(chat_models) and not missing,
        real_payload=latest.get("multi_model"),
        real_fresh=real_fresh,
    )
    workflow_security = _report_rollup(latest.get("workflow"), workflow_fresh)
    hardened_go_live, hardened_reason = _hardened_rollup(
        latest.get("go_live"), go_live_fresh, readiness=readiness, settings=settings
    )

    report = DailyDriverReport(
        generated_at=utc_now_iso(),
        config_fingerprint=fingerprint,
        runtime_backend=backend,
        core_real_model=core_real_model,
        workflow_security=workflow_security,
        hardened_go_live=hardened_go_live,
        hardened_reason=hardened_reason,
        overall=_worst([check.status for check in checks]),
        checks=checks,
    )
    report.next_commands = _next_commands(report)
    return report


def _config_errors(home: Path) -> list[str]:
    # Imported lazily to avoid a heavy import at module load.
    from april_common.config_validation import validate_configuration

    try:
        return list(validate_configuration(home))
    except Exception as exc:  # pragma: no cover - defensive
        return [str(exc)]


def _safe_settings(home: Path) -> AprilSettings | None:
    try:
        return load_settings(root=home)
    except ConfigError:
        return None


def _report_check(
    name: str,
    item: tuple[Path, dict] | None,
    fingerprint: str | None,
    next_command: str,
) -> tuple[ReportFreshness | None, DailyDriverCheck]:
    if item is None:
        return None, DailyDriverCheck(
            name=name, status="not_run", detail="no report found", next_command=next_command
        )
    path, payload = item
    report_type = classify_report(payload)
    fresh = freshness_from_payload(
        payload, report_type=report_type, current_fingerprint=fingerprint, basename=path.name
    )
    status_value = _report_status(payload) or "unknown"
    failed = status_value in {"fail", "failed"}
    if failed:
        status: DailyStatus = "blocker"
        detail = f"{path.name}: {status_value}"
    elif fresh.stale:
        status = "warning"
        detail = f"{path.name}: {status_value}, stale ({fresh.stale_reason})"
    else:
        status = "ready"
        detail = f"{path.name}: {status_value}, age {fresh.age_human or 'unknown'}"
    return fresh, DailyDriverCheck(
        name=name,
        status=status,
        detail=detail,
        next_command=next_command if status != "ready" else None,
    )


def _token_check(readiness: ReadinessReport) -> DailyDriverCheck:
    statuses = {readiness.api_token_status, readiness.runtime_token_status}
    if statuses <= {"configured"}:
        return DailyDriverCheck(
            name="token hardening", status="ready", detail="API/runtime tokens configured"
        )
    if "placeholder-insecure" in statuses:
        detail = "insecure placeholder tokens are active"
    elif "default-development" in statuses:
        detail = "development tokens are still active"
    else:
        detail = "a loopback token is not configured"
    return DailyDriverCheck(
        name="token hardening", status="warning", detail=detail, next_command=_SETUP_TOKENS
    )


def _embedding_check(settings: AprilSettings | None) -> DailyDriverCheck:
    if settings is None:
        return DailyDriverCheck(
            name="embedding provider", status="warning", detail="settings unavailable"
        )
    provider = settings.memory.embedding_provider
    if provider == "runtime-local":
        model_id = settings.memory.embedding_model_id
        if model_id:
            return DailyDriverCheck(
                name="embedding provider",
                status="ready",
                detail=f"runtime-local (embedding model: {model_id})",
            )
        return DailyDriverCheck(
            name="embedding provider",
            status="warning",
            detail="runtime-local requested but no embedding model id is configured",
            next_command=_SETUP_EMBEDDINGS,
        )
    # hashed-token is the safe offline default, but not strong semantic memory and
    # not the hardened path.
    return DailyDriverCheck(
        name="embedding provider",
        status="warning",
        detail="hashed-token (safe offline default; runtime-local recommended for semantic memory)",
        next_command=_SETUP_EMBEDDINGS,
    )


def _vector_index_check(home: Path, settings: AprilSettings | None) -> DailyDriverCheck:
    import json

    if settings is None:
        return DailyDriverCheck(
            name="vector index compatibility", status="warning", detail="settings unavailable"
        )
    metadata_path = settings.vector_index_path / "metadata.json"
    if not metadata_path.exists():
        return DailyDriverCheck(
            name="vector index compatibility",
            status="ready",
            detail="no index yet (built on first memory write)",
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DailyDriverCheck(
            name="vector index compatibility",
            status="warning",
            detail="index metadata is unreadable",
            next_command=_MEMORY_REINDEX,
        )
    persisted = metadata.get("provider") if isinstance(metadata, dict) else None
    active = settings.memory.embedding_provider
    if persisted is not None and persisted != active:
        return DailyDriverCheck(
            name="vector index compatibility",
            status="warning",
            detail=f"index built with '{persisted}' but provider is '{active}' (reindex required)",
            next_command=_MEMORY_REINDEX,
        )
    return DailyDriverCheck(
        name="vector index compatibility",
        status="ready",
        detail=f"index matches active provider ({active})",
    )


def _voice_check(
    readiness: ReadinessReport, latest: dict[str, tuple[Path, dict]]
) -> DailyDriverCheck:
    # Voice is optional and never a blocker. Live verification status is read from
    # redacted reports; offline artifact presence supplies the rest. No device or
    # microphone is ever touched here.
    if not readiness.voice_enabled:
        return DailyDriverCheck(
            name="voice milestone", status="ready", detail="disabled (optional)"
        )
    if "wake_word_live" in latest:
        return DailyDriverCheck(name="voice milestone", status="ready", detail="wake_live_verified")
    if "voice_live" in latest:
        return DailyDriverCheck(name="voice milestone", status="ready", detail="live_verified")
    artifacts = {artifact.name: artifact for artifact in readiness.voice_artifacts}

    def present(name: str) -> bool:
        artifact = artifacts.get(name)
        return bool(artifact and artifact.configured and artifact.exists)

    push_to_talk = (
        present("whisper.cpp binary")
        and present("whisper model")
        and present("piper binary")
        and present("piper voice model")
    )
    if not push_to_talk:
        return DailyDriverCheck(
            name="voice milestone",
            status="warning",
            detail="enabled but not configured (push-to-talk artifacts missing)",
            next_command="run april voice doctor",
        )
    milestone = "wake_word_ready" if present("wake-word model") else "push_to_talk_ready"
    return DailyDriverCheck(
        name="voice milestone",
        status="warning",
        detail=f"{milestone} (run voice verify-live to confirm)",
        next_command="run april voice verify-live --report data/verification/voice-live.json",
    )


def _core_rollup(
    *,
    config_valid: bool,
    readiness: ReadinessReport,
    chat_models_present: bool,
    real_payload: tuple[Path, dict] | None,
    real_fresh: ReportFreshness | None,
) -> DailyStatus:
    if not config_valid or readiness.runtime_is_fake or not readiness.llama_cpp_python_available:
        return "blocker"
    if not chat_models_present:
        return "blocker"
    if real_payload is None:
        return "not_run"
    status_value = _report_status(real_payload[1]) or "unknown"
    if status_value in {"fail", "failed"}:
        return "blocker"
    if real_fresh is not None and real_fresh.stale:
        return "warning"
    if status_value == "pass" or bool(real_payload[1].get("real_model_verified")):
        return "ready"
    return "warning"


def _report_rollup(
    payload_item: tuple[Path, dict] | None, fresh: ReportFreshness | None
) -> DailyStatus:
    if payload_item is None:
        return "not_run"
    status_value = _report_status(payload_item[1]) or "unknown"
    if status_value in {"fail", "failed"}:
        return "blocker"
    if fresh is not None and fresh.stale:
        return "warning"
    if status_value == "pass":
        return "ready"
    return "warning"


def _hardened_rollup(
    payload_item: tuple[Path, dict] | None,
    fresh: ReportFreshness | None,
    *,
    readiness: ReadinessReport,
    settings: AprilSettings | None,
) -> tuple[DailyStatus, str | None]:
    reasons: list[str] = []
    if {readiness.api_token_status, readiness.runtime_token_status} - {"configured"}:
        reasons.append("development tokens")
    if settings is not None and settings.memory.embedding_provider != "runtime-local":
        reasons.append("hashed-token embeddings")
    reason = ", ".join(reasons) or None
    if payload_item is None:
        return "not_run", reason
    payload = payload_item[1]
    if str(payload.get("final_status")) == "fail":
        return "blocker", reason
    if fresh is not None and fresh.stale:
        return "warning", fresh.stale_reason or reason
    if bool(payload.get("hardened_go_live_ready")):
        return "ready", None
    # Surface the report's own hardening warnings when the live config has not
    # already supplied a more specific reason.
    hardening = payload.get("hardening_warnings")
    if not reason and isinstance(hardening, list) and hardening:
        reason = ", ".join(str(item) for item in hardening)
    return "warning", reason


def _next_commands(report: DailyDriverReport) -> list[str]:
    # Recommended order, de-duplicated, derived from the failing/warning checks.
    ordered: list[str] = []

    def add(command: str | None) -> None:
        if command and command not in ordered:
            ordered.append(command)

    for check in report.checks:
        if check.status in {"blocker", "warning", "not_run"}:
            add(check.next_command)
    return ordered
