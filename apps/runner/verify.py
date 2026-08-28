from __future__ import annotations

# Compatibility facade: tests and integrations intentionally import and
# monkeypatch these re-exported collaborators through apps.runner.verify.
# ruff: noqa: F401
import os
import platform
import socket
from pathlib import Path

import httpx as httpx

from apps.runner.mac_report import ReportThresholds
from apps.runner.verification.local_checks import (
    run_local_sandbox_verification,
    run_local_security_integrity_verification,
)
from apps.runner.verification.models import ModelBenchmark, RealModelVerifier
from apps.runner.verification.multi_model import AllConfiguredModelsVerifier
from apps.runner.verification.planning import (
    plan_multi_model_verification as _plan_multi_model_verification,
)
from apps.runner.verification.planning import skipped_result_for
from apps.runner.verification.reports import (
    brain_decision_after_marker,
    build_workflow_report,
    chat_result_from_response,
    latest_brain_decision_marker,
    write_workflow_report,
)
from apps.runner.verification.reports import (
    json_object_candidates as _json_object_candidates,
)
from apps.runner.verification.reports import (
    safe_workflow_report_detail as _safe_workflow_report_detail,
)
from apps.runner.verification.services import LauncherVerifier
from apps.runner.verification.target_mac import TargetMacValidator
from apps.runner.verification.types import (
    BenchmarkResult,
    ModelPlanEntry,
    VerifyCheck,
)
from apps.runner.verification.workflow import RealWorkflowVerifier, WorkflowVerifier
from april_common.credentials import CredentialKey, FileCredentialStore
from april_common.process_environment import ProcessCategory
from april_common.process_runner import run_restricted_process_sync
from april_common.service_health import ServiceHealthResult
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


def run_fake_verification(
    home: Path,
    *,
    development_unsandboxed_override: bool = False,
) -> list[VerifyCheck]:
    verifier = LauncherVerifier(
        home=home,
        development_unsandboxed_override=development_unsandboxed_override,
    )
    return [*verifier.run(), *run_local_sandbox_verification(home)]


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
    """Compatibility wrapper around the focused verification planner."""
    available = _llama_cpp_installed() if llama_available is None else llama_available
    return _plan_multi_model_verification(home, llama_available=available)


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
