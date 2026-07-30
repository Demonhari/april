from __future__ import annotations

import json
import os
import platform
import re
import socket
import sqlite3
from pathlib import Path
from typing import Any

import httpx as httpx

from apps.runner.mac_report import (
    ReportThresholds,
    environment_snapshot,
    quantization_from_basename,
    redact_reason,
)
from apps.runner.multi_model_report import (
    PerModelResult,
)
from apps.runner.verification.models import ModelBenchmark, RealModelVerifier
from apps.runner.verification.multi_model import AllConfiguredModelsVerifier
from apps.runner.verification.services import LauncherVerifier
from apps.runner.verification.target_mac import TargetMacValidator
from apps.runner.verification.types import (
    BenchmarkResult,
    MissingChatResultError,
    ModelPlanEntry,
    VerifyCheck,
    VerifyStatus,
    WorkflowReportCheck,
    WorkflowVerificationReport,
)
from apps.runner.verification.workflow import RealWorkflowVerifier, WorkflowVerifier
from april_common.audit import audit_logger_for_settings
from april_common.credentials import CredentialKey, FileCredentialStore
from april_common.errors import ConfigError
from april_common.process_environment import ProcessCategory
from april_common.process_runner import run_restricted_process_sync
from april_common.process_sandbox import SandboxBackend, sandbox_capabilities
from april_common.service_health import ServiceHealthResult
from april_common.settings import load_settings
from april_common.token_setup import legacy_plaintext_credentials_detected
from services.april_runtime.model_registry import ModelRegistry
from services.memory.database import connect_sqlite
from services.memory.maintenance import check_database
from services.voice.health import query_audio_devices as query_audio_devices
from services.voice.health import voice_doctor as voice_doctor


def _verification_health_failure(
    url: str,
    api_url: str,
    result: ServiceHealthResult,
) -> str:
    if url.startswith(api_url):
        return "Core API process is not running or its liveness endpoint is unreachable."
    if result.reason == "authentication_rejected":
        return "Runtime authentication was rejected."
    if result.reason == "endpoint_not_found":
        return "Runtime health endpoint returned 404; check the configured endpoint."
    return "Runtime is not reachable."


def run_fake_verification(home: Path) -> list[VerifyCheck]:
    verifier = LauncherVerifier(home=home)
    return [*verifier.run(), *run_local_sandbox_verification(home)]


def run_local_sandbox_verification(home: Path) -> list[VerifyCheck]:
    try:
        settings = load_settings(root=home)
    except (ConfigError, RuntimeError) as exc:
        return [
            VerifyCheck(
                name="Tool Worker sandbox capability",
                ok=False,
                detail=f"unavailable ({type(exc).__name__})",
            )
        ]
    report = sandbox_capabilities(
        environment=settings.environment,
        development_override=settings.workers.development_unsandboxed_override,
    )
    backend_available = report.backend is not SandboxBackend.UNAVAILABLE
    production = settings.environment == "production"
    unavailable_status: VerifyStatus = "fail" if production else "skip"
    return [
        VerifyCheck(
            "sandbox backend",
            backend_available or not production,
            report.backend.value,
            status="pass" if backend_available else unavailable_status,
        ),
        VerifyCheck(
            "sandbox network denial",
            report.network_denial_available or not production,
            "available" if report.network_denial_available else "unavailable",
            status="pass" if report.network_denial_available else unavailable_status,
        ),
        VerifyCheck(
            "sandbox filesystem policy",
            report.filesystem_policy_available or not production,
            "available" if report.filesystem_policy_available else "unavailable",
            status="pass" if report.filesystem_policy_available else unavailable_status,
        ),
        VerifyCheck(
            "sandbox production fail closed",
            report.production_fail_closed,
            "enabled" if report.production_fail_closed else "disabled",
        ),
        VerifyCheck(
            "sandbox development override",
            not report.development_override_enabled or not production,
            report.warning or "disabled",
            status="skip" if report.development_override_enabled else "pass",
        ),
    ]


def run_local_security_integrity_verification(home: Path) -> list[VerifyCheck]:
    """Explicit local Phase 4B checks; never exposes credential values or payloads."""
    try:
        settings = load_settings(root=home)
    except (ConfigError, RuntimeError) as exc:
        return [
            VerifyCheck(
                name="security configuration",
                ok=False,
                detail=f"unavailable ({type(exc).__name__})",
            )
        ]
    store_name: str = settings.security.credential_store
    if store_name == "auto":
        store_name = (
            "macos-keychain"
            if settings.environment == "production" and platform.system() == "Darwin"
            else "legacy-development-default"
        )
    legacy = legacy_plaintext_credentials_detected(settings.home)
    audit_result = audit_logger_for_settings(settings).verify()
    database = check_database(settings.database_path, home=settings.home, full=False)
    backup = database.last_successful_backup
    backup_detail = str(backup.get("creation_timestamp", "known")) if backup else "none recorded"
    return [
        *run_local_sandbox_verification(home),
        VerifyCheck("credential store selected", True, store_name),
        VerifyCheck(
            "API credential available",
            bool(settings.api.token),
            "available" if settings.api.token else "missing",
        ),
        VerifyCheck(
            "Runtime credential available",
            bool(settings.runtime.token),
            "available" if settings.runtime.token else "missing",
        ),
        VerifyCheck(
            "legacy plaintext credential",
            not legacy,
            "detected; run security credentials migrate" if legacy else "not detected",
        ),
        VerifyCheck(
            "audit chain",
            not audit_result.corrupt,
            audit_result.status,
        ),
        VerifyCheck(
            "database quick_check",
            database.quick_check == "ok",
            database.quick_check,
        ),
        VerifyCheck(
            "database foreign keys",
            database.foreign_key_consistent,
            (
                "ok"
                if database.foreign_key_consistent
                else f"{database.foreign_key_violations} violation(s)"
            ),
        ),
        VerifyCheck("database WAL state", database.journal_mode == "wal", database.journal_mode),
        VerifyCheck("last successful backup", True, backup_detail),
    ]


def run_workflow_verification(
    home: Path,
    *,
    real_model: bool = False,
    model_path: Path | None = None,
    max_output_tokens: int = 32,
    timeout: float = 180.0,
) -> list[VerifyCheck]:
    if real_model:
        configured_path = model_path or (
            Path(os.environ["APRIL_TEST_GGUF_PATH"])
            if os.environ.get("APRIL_TEST_GGUF_PATH")
            else None
        )
        if configured_path is None:
            return [
                VerifyCheck(
                    name="real workflow planning route",
                    ok=False,
                    detail="APRIL_TEST_GGUF_PATH or --real-model path is required.",
                )
            ]
        return RealWorkflowVerifier(
            home=home,
            model_path=configured_path,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
        ).run()
    return WorkflowVerifier(home=home).run()


def run_real_model_verification(
    home: Path,
    model_path: Path,
    *,
    max_output_tokens: int = 32,
    timeout: float = 180.0,
) -> list[VerifyCheck]:
    if not _llama_cpp_installed():
        return [
            VerifyCheck(
                name="llama-cpp-python installed",
                ok=False,
                detail="pip install -e '.[runtime]'",
            )
        ]
    verifier = RealModelVerifier(
        home=home,
        model_path=model_path,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
    )
    return verifier.run()


def run_target_mac_validation(
    home: Path,
    *,
    model_path: Path | None = None,
    require_real_model: bool = False,
    max_output_tokens: int = 32,
    timeout: float = 180.0,
) -> list[VerifyCheck]:
    validator = TargetMacValidator(
        home=home,
        model_path=model_path,
        require_real_model=require_real_model,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
    )
    return validator.run()


def _infer_chat_format_from_basename(basename: str) -> str:
    """Best-effort chat-format family from a GGUF *basename*, defaulting to the
    always-supported ``generic`` template.

    Used only by the standalone single-file verifier/benchmark, which fabricate a
    model entry with no operator-set ``chat_format``. The runtime's resolver only
    infers from ``model.name`` (here a fixed sentinel), so without this it would
    raise "Unsupported chat template" for every supplied model. ``generic`` always
    produces a usable prompt, so an arbitrary model still gets a structural
    load/chat/stream/unload smoke; ``granite``/``qwen`` are used when recognised.
    """
    normalized = basename.casefold()
    if "granite" in normalized:
        return "granite"
    if "qwen" in normalized:
        return "qwen"
    return "generic"


def plan_multi_model_verification(
    home: Path, *, llama_available: bool | None = None
) -> list[ModelPlanEntry]:
    """Inspect ``configs/models.yaml`` and decide which real models can be run.

    Never downloads anything; only reads local configuration and checks local
    file existence/readability. Embedding-role models are reported as skipped
    because they are verified through ``run april memory reindex``, not chat.
    """
    available = _llama_cpp_installed() if llama_available is None else llama_available
    registry = ModelRegistry.from_file(home / "configs" / "models.yaml", root=home)
    entries: list[ModelPlanEntry] = []
    for model in registry.list():
        path = model.resolved_path(registry.root)
        if model.backend != "llama_cpp":
            reason = f"Backend {model.backend} is not a real GGUF backend."
            entries.append(ModelPlanEntry(model=model, path=path, available=False, reason=reason))
        elif model.role == "embedding":
            entries.append(
                ModelPlanEntry(
                    model=model,
                    path=path,
                    available=False,
                    reason="Embedding model is verified via `run april memory reindex`, not chat.",
                )
            )
        elif not available:
            entries.append(
                ModelPlanEntry(
                    model=model,
                    path=path,
                    available=False,
                    reason="llama-cpp-python is not installed (pip install -e '.[runtime]').",
                )
            )
        elif not path.exists():
            entries.append(
                ModelPlanEntry(
                    model=model, path=path, available=False, reason=f"Missing model file: {path}"
                )
            )
        elif not os.access(path, os.R_OK):
            entries.append(
                ModelPlanEntry(
                    model=model, path=path, available=False, reason=f"Not readable: {path}"
                )
            )
        else:
            entries.append(ModelPlanEntry(model=model, path=path, available=True, reason=None))
    return entries


def skipped_result_for(entry: ModelPlanEntry) -> PerModelResult:
    """A redacted per-model result for a model that was not exercised."""
    return PerModelResult(
        model_id=entry.model.id,
        role=entry.model.role,
        backend=entry.model.backend,
        path_basename=entry.path_basename,
        quantization=quantization_from_basename(entry.path_basename),
        available=False,
        skipped_reason=entry.reason,
    )


def run_all_configured_models_verification(
    home: Path,
    *,
    require_real_model: bool = False,
    max_output_tokens: int = 32,
    timeout: float = 180.0,
    thresholds: ReportThresholds | None = None,
    candidate_adapter_model_id: str | None = None,
    candidate_adapter_path: Path | None = None,
) -> AllConfiguredModelsVerifier:
    verifier = AllConfiguredModelsVerifier(
        home=home,
        require_real_model=require_real_model,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
        thresholds=thresholds,
        candidate_adapter_model_id=candidate_adapter_model_id,
        candidate_adapter_path=candidate_adapter_path,
    )
    verifier.run()
    return verifier


def latest_brain_decision_marker(database: Path) -> int:
    if not database.exists():
        return 0
    try:
        with connect_sqlite(database) as conn:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(rowid), 0)
                FROM conversation_events
                WHERE event_type = 'brain_decision'
                """
            ).fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0]) if row is not None and row[0] is not None else 0


def brain_decision_after_marker(database: Path, marker: int) -> dict[str, Any]:
    if not database.exists():
        return {}
    try:
        with connect_sqlite(database) as conn:
            row = conn.execute(
                """
                SELECT payload_json
                FROM conversation_events
                WHERE event_type = 'brain_decision' AND rowid > ?
                ORDER BY rowid DESC
                LIMIT 1
                """,
                (marker,),
            ).fetchone()
    except sqlite3.Error:
        return {}
    if row is None:
        return {}
    try:
        payload = json.loads(str(row[0]))
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_object_candidates(text: str) -> list[dict[str, Any]]:
    """Return every parseable JSON object embedded in model text.

    Real local models often wrap valid JSON in markdown fences, short reasoning
    preambles, or prompt echoes. Verification should not treat that wrapper text
    as a model-runtime failure when a valid object with the required shape is
    present.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    candidates: list[dict[str, Any]] = []
    depth = 0
    start: int | None = None
    in_string = False
    escape = False
    for index, char in enumerate(stripped):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                raw = re.sub(r",(\s*[}\]])", r"\1", stripped[start : index + 1])
                try:
                    parsed = json.loads(raw)
                except ValueError:
                    pass
                else:
                    if isinstance(parsed, dict):
                        candidates.append(parsed)
                start = None
            elif depth < 0:
                depth = 0
                start = None
    return candidates


def chat_result_from_response(
    response: Any, *, context: str, snippet_chars: int = 240
) -> dict[str, Any]:
    """Validate a /chat result and retain a bounded body clue on failure."""
    try:
        payload = response.json()
    except (TypeError, ValueError):
        payload = None
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict):
        raw = " ".join(str(getattr(response, "text", "")).split())
        snippet = raw[: max(0, snippet_chars)] or "<empty body>"
        raise MissingChatResultError(f"{context} response missing result; body={snippet}")
    response.raise_for_status()
    return result


def build_workflow_report(
    checks: list[VerifyCheck],
    *,
    real_model_requested: bool,
    timeout_seconds: float | None = None,
    max_output_tokens: int | None = None,
    config_fingerprint: str | None = None,
) -> WorkflowVerificationReport:
    failed = [check for check in checks if not check.ok]
    real_model_exercised = real_model_requested and any(
        check.name == "real workflow planning route" and check.ok for check in checks
    )
    real_model_verified = real_model_exercised and not failed
    rendered = [
        WorkflowReportCheck(
            name=check.name,
            ok=check.ok,
            status=check.status or ("pass" if check.ok else "fail"),
            detail=_safe_workflow_report_detail(check.detail),
        )
        for check in checks
    ]
    return WorkflowVerificationReport(
        generated_at=environment_snapshot().generated_at,
        config_fingerprint=config_fingerprint,
        summary="pass" if not failed else "fail",
        real_model_verified=real_model_verified,
        real_model_exercised=real_model_exercised,
        checks=rendered,
        checks_failed=len(failed),
        check_failures=[check.name for check in failed],
        timeout_seconds=timeout_seconds if real_model_requested else None,
        max_output_tokens=max_output_tokens if real_model_requested else None,
    )


def write_workflow_report(report: WorkflowVerificationReport, path: Path) -> Path:
    resolved = path.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        report.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8"
    )
    return resolved


def _safe_workflow_report_detail(detail: str) -> str:
    lower = detail.lower()
    if "decision_summary" in lower:
        return "decision_summary redacted"
    sensitive_markers = (
        "prompt",
        "transcript",
        "token",
        "authorization",
        "bearer",
        "raw_tool_args",
        "tool args",
    )
    if any(marker in lower for marker in sensitive_markers):
        return "sensitive detail redacted"
    return redact_reason(detail)[:240]


def run_model_benchmark(
    home: Path,
    model_path: Path,
    *,
    prompt: str,
    runs: int,
    max_output_tokens: int,
    keep_loaded: bool,
) -> list[BenchmarkResult]:
    if not _llama_cpp_installed():
        return [
            BenchmarkResult(
                run_index=1,
                ok=False,
                detail="llama-cpp-python is missing. Install with: pip install -e '.[runtime]'",
            )
        ]
    benchmark = ModelBenchmark(
        home=home,
        model_path=model_path,
        prompt=prompt,
        runs=runs,
        max_output_tokens=max_output_tokens,
        keep_loaded=keep_loaded,
    )
    return benchmark.run()


def _llama_cpp_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("llama_cpp") is not None


def _verification_credential_environment(
    *,
    verify_home: Path,
    temporary_root: Path,
    api_token: str,
    runtime_token: str,
) -> dict[str, str]:
    """Provision ephemeral test credentials without putting values in child env."""
    credential_path = temporary_root / "verification-credentials.json"
    store = FileCredentialStore(credential_path, repository_root=verify_home)
    store.set(CredentialKey.API_TOKEN, api_token)
    store.set(CredentialKey.RUNTIME_TOKEN, runtime_token)
    return {
        "APRIL_CREDENTIAL_STORE": "file",
        "APRIL_CREDENTIAL_FILE_PATH": str(credential_path),
        "APRIL_API_CREDENTIAL_ID": CredentialKey.API_TOKEN.value,
        "APRIL_RUNTIME_CREDENTIAL_ID": CredentialKey.RUNTIME_TOKEN.value,
    }


def _process_rss_bytes(pid: int | None) -> int | None:
    if pid is None:
        return None
    completed = run_restricted_process_sync(
        ["ps", "-o", "rss=", "-p", str(pid)],
        cwd=Path.cwd(),
        category=ProcessCategory.VERIFICATION_SUBPROCESS,
        timeout_seconds=5,
        max_stdout_bytes=4096,
        max_stderr_bytes=4096,
    )
    if completed.returncode is None or completed.returncode != 0:
        return None
    try:
        return int(completed.stdout.strip().split()[0]) * 1024
    except (ValueError, IndexError):
        return None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _git(cwd: Path, *args: str) -> None:
    result = run_restricted_process_sync(
        ["git", *args],
        cwd=cwd,
        category=ProcessCategory.GIT,
        timeout_seconds=15,
        max_stdout_bytes=100_000,
        max_stderr_bytes=100_000,
    )
    if result.returncode != 0:
        raise RuntimeError("Verification Git command failed.")
