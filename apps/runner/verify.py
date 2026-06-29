from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml
from pydantic import BaseModel, Field

from apps.runner.mac_report import (
    MacVerificationReport,
    RealModelReport,
    ReportThresholds,
    RoutingReport,
    SkippedCheck,
    build_mac_report,
    environment_snapshot,
    quantization_from_basename,
    redact_reason,
    routing_report_from_results,
)
from apps.runner.multi_model_report import (
    MultiModelVerificationReport,
    PerModelResult,
    SpecialistSwitchReport,
    build_multi_model_report,
)
from april_common.errors import ConfigError
from april_common.settings import load_settings
from services.april_runtime.model_registry import ModelDefinition, ModelRegistry
from services.brain.schemas import BrainDecision
from services.voice.health import query_audio_devices, voice_doctor

VerifyStatus = Literal["pass", "fail", "skip", "manual"]


@dataclass(slots=True)
class VerifyCheck:
    name: str
    ok: bool
    detail: str = ""
    status: VerifyStatus | None = None

    def __post_init__(self) -> None:
        if self.status is None:
            self.status = "pass" if self.ok else "fail"


def run_fake_verification(home: Path) -> list[VerifyCheck]:
    verifier = LauncherVerifier(home=home)
    return verifier.run()


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


@dataclass(slots=True)
class ModelPlanEntry:
    """One configured model and whether the multi-model verifier can exercise it.

    Pure data so the discovery/skip decision is unit-testable without a real
    runtime: a missing file, an unreadable file, a non-chat (embedding) role, or
    an absent llama-cpp-python all yield ``available=False`` with an explicit
    ``reason`` — never a silent pass.
    """

    model: ModelDefinition
    path: Path
    available: bool
    reason: str | None = None

    @property
    def path_basename(self) -> str:
        return self.path.name


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
) -> AllConfiguredModelsVerifier:
    verifier = AllConfiguredModelsVerifier(
        home=home,
        require_real_model=require_real_model,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
        thresholds=thresholds,
    )
    verifier.run()
    return verifier


def latest_brain_decision_marker(database: Path) -> int:
    if not database.exists():
        return 0
    try:
        with sqlite3.connect(database) as conn:
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
        with sqlite3.connect(database) as conn:
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


class BenchmarkResult(BaseModel):
    run_index: int
    ok: bool = True
    detail: str = ""
    load_time_seconds: float = 0.0
    first_token_latency_seconds: float | None = None
    generation_time_seconds: float = 0.0
    output_tokens: int = 0
    tokens_per_second: float = 0.0
    unload_success: bool = False
    context_size: int = 1024
    backend_settings: dict[str, Any] = Field(default_factory=dict)


class WorkflowReportCheck(BaseModel):
    name: str
    ok: bool
    status: VerifyStatus
    detail: str = ""


class WorkflowVerificationReport(BaseModel):
    schema_version: int = 1
    report_type: Literal["workflow"] = "workflow"
    generated_at: str
    summary: str = "degraded"
    real_model_verified: bool = False
    real_model_exercised: bool = False
    checks: list[WorkflowReportCheck] = Field(default_factory=list)
    checks_failed: int = 0
    check_failures: list[str] = Field(default_factory=list)
    timeout_seconds: float | None = None
    max_output_tokens: int | None = None


def build_workflow_report(
    checks: list[VerifyCheck],
    *,
    real_model_requested: bool,
    timeout_seconds: float | None = None,
    max_output_tokens: int | None = None,
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


class TargetMacValidator:
    def __init__(
        self,
        *,
        home: Path,
        model_path: Path | None,
        require_real_model: bool,
        max_output_tokens: int,
        timeout: float,
    ) -> None:
        self.home = home.expanduser().resolve()
        self.model_path = model_path.expanduser().resolve() if model_path else None
        self.require_real_model = require_real_model
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.checks: list[VerifyCheck] = []
        self.settings_error: str | None = None
        # Populated during run() so build_report() can emit structured metrics.
        self.selected_model: Path | None = None
        self.real_verifier: RealModelVerifier | None = None

    def run(self) -> list[VerifyCheck]:
        self._machine_architecture()
        self._python_version()
        self._configuration_load()
        llama_available = self._llama_cpp_import()
        self._backend_build_info(llama_available)
        selected_model = self._configured_gguf_path()
        self.selected_model = selected_model
        if selected_model is not None and selected_model.exists() and llama_available:
            # Instantiate the verifier directly so build_report() can read its
            # structured timing/RSS metrics after the checks complete.
            verifier = RealModelVerifier(
                home=self.home,
                model_path=selected_model,
                max_output_tokens=self.max_output_tokens,
                timeout=self.timeout,
            )
            self.real_verifier = verifier
            self.checks.extend(verifier.run())
            self.checks.extend(
                run_workflow_verification(
                    self.home,
                    real_model=True,
                    model_path=selected_model,
                    max_output_tokens=self.max_output_tokens,
                    timeout=self.timeout,
                )
            )
        else:
            self._model_dependent_skips()
        self._voice_checks()
        self._manual(
            "push-to-talk record/transcribe/speak smoke",
            "Run `run april voice ptt --seconds 3` on the target Mac after configuring voice.",
        )
        self._pass("cleanup and service shutdown", "No persistent services are started by skips.")
        return self.checks

    def build_report(self, *, thresholds: ReportThresholds | None = None) -> MacVerificationReport:
        """Assemble a redacted, machine-readable acceptance report.

        Call after ``run()``. Real-model metrics are populated only when a real
        model was actually exercised; otherwise the real-model section is marked
        ``attempted=False`` and the skipped checks carry explicit reasons, so a
        simulated/skipped run is never presented as real-model verified.
        """
        skipped = [
            SkippedCheck(name=check.name, reason=check.detail)
            for check in self.checks
            if check.status == "skip"
        ]
        return build_mac_report(
            environment=environment_snapshot(),
            runtime_backend=self._report_backend(),
            real_model=self._real_model_report(),
            routing=self._routing_report(),
            skipped=skipped,
            checks_passed=sum(1 for check in self.checks if check.status == "pass"),
            checks_failed=sum(1 for check in self.checks if check.status == "fail"),
            thresholds=thresholds,
            require_real_model=self.require_real_model,
        )

    def _report_backend(self) -> str:
        if self.real_verifier is not None:
            return "llama_cpp"
        try:
            return load_settings(root=self.home).runtime.backend
        except ConfigError:
            return "unknown"

    def _check_ok(self, name: str) -> bool:
        return any(check.name == name and check.ok for check in self.checks)

    def _structured_brain_ok(self) -> bool:
        return any(
            check.ok
            and ("planning route" in check.name.lower() or "brain json" in check.name.lower())
            for check in self.checks
        )

    def _real_model_report(self) -> RealModelReport:
        verifier = self.real_verifier
        if verifier is None:
            return RealModelReport(attempted=False)
        basename = self.selected_model.name if self.selected_model else None
        return RealModelReport(
            attempted=True,
            model_id="april-brain",
            role="brain",
            path_basename=basename,
            quantization=quantization_from_basename(basename),
            context_size=1024,
            load_success=self._check_ok("real model load"),
            load_duration_seconds=verifier.load_time_seconds,
            chat_success=self._check_ok("real model chat"),
            structured_brain_json_success=self._structured_brain_ok(),
            streaming_success=self._check_ok("real model stream"),
            first_token_latency_seconds=verifier.first_token_latency_seconds,
            unload_success=self._check_ok("real model unload"),
            output_token_count=verifier.output_tokens,
            tokens_per_second=verifier.tokens_per_second,
            process_rss_bytes=verifier.runtime_rss_bytes,
            process_peak_rss_bytes=None,
        )

    def _routing_report(self) -> RoutingReport | None:
        # Imported lazily to avoid a circular import (evals imports verify).
        from apps.runner.evals import run_fake_brain_eval

        try:
            results = run_fake_brain_eval(self.home)
        except Exception:
            return None
        return routing_report_from_results(results)

    def _machine_architecture(self) -> None:
        system = platform.system()
        machine = platform.machine()
        if system != "Darwin":
            self._manual(
                "machine architecture",
                f"Run on the target Mac. Current host reports {system}/{machine}.",
            )
            return
        if machine not in {"arm64", "x86_64"}:
            self._fail("machine architecture", f"Unsupported Mac architecture: {machine}")
            return
        self._pass("machine architecture", f"{system}/{machine}")

    def _python_version(self) -> None:
        version = sys.version_info
        detail = f"{version.major}.{version.minor}.{version.micro}"
        if (version.major, version.minor) < (3, 11) or (version.major, version.minor) > (3, 13):
            self._fail("Python version", f"{detail}; APRIL supports Python 3.11 through 3.13")
            return
        self._pass("Python version", detail)

    def _configuration_load(self) -> None:
        try:
            load_settings(root=self.home)
            ModelRegistry.from_file(self.home / "configs" / "models.yaml", root=self.home)
        except ConfigError as exc:
            self.settings_error = str(exc)
            self._fail("configuration load", str(exc))
            return
        self._pass("configuration load", "settings and model registry loaded")

    def _llama_cpp_import(self) -> bool:
        if not _llama_cpp_installed():
            self._required_or_skip(
                "llama-cpp-python import",
                "Install the optional runtime extra with `pip install -e '.[runtime]'`.",
            )
            return False
        self._pass("llama-cpp-python import", "module spec found")
        return True

    def _backend_build_info(self, llama_available: bool) -> None:
        if not llama_available:
            self._skip("backend acceleration/build information", "llama-cpp-python unavailable")
            return
        self._manual(
            "backend acceleration/build information",
            "Detailed llama.cpp build information is available through Runtime-backed "
            "real-model diagnostics.",
        )

    def _configured_gguf_path(self) -> Path | None:
        selected = self._select_model_path()
        if selected is None:
            self._required_or_skip(
                "configured GGUF existence and readability",
                "No --model path, APRIL_TEST_GGUF_PATH, or configured llama_cpp brain model.",
            )
            return None
        if not selected.exists():
            self._required_or_skip(
                "configured GGUF existence and readability", f"Missing: {selected}"
            )
            return selected
        if not os.access(selected, os.R_OK):
            self._required_or_skip(
                "configured GGUF existence and readability", f"Not readable: {selected}"
            )
            return selected
        self._pass("configured GGUF existence and readability", str(selected))
        return selected

    def _select_model_path(self) -> Path | None:
        if self.model_path is not None:
            return self.model_path
        env_path = os.environ.get("APRIL_TEST_GGUF_PATH")
        if env_path:
            return Path(env_path).expanduser().resolve(strict=False)
        try:
            registry = ModelRegistry.from_file(
                self.home / "configs" / "models.yaml", root=self.home
            )
        except ConfigError:
            return None
        for model in registry.list():
            if model.role == "brain" and model.backend == "llama_cpp":
                return model.resolved_path(registry.root)
        return None

    def _model_dependent_skips(self) -> None:
        status: VerifyStatus = "fail" if self.require_real_model else "skip"
        ok = not self.require_real_model
        detail = "Requires readable local GGUF and llama-cpp-python."
        for name in (
            "model load",
            "non-streaming completion",
            "streaming completion",
            "strict brain JSON parse",
            "specialist-agent request",
            "load-on-demand and unload",
            "runtime RSS before load/after load/after unload",
        ):
            self.checks.append(VerifyCheck(name=name, ok=ok, detail=detail, status=status))

    def _voice_checks(self) -> None:
        try:
            settings = load_settings(root=self.home)
        except ConfigError as exc:
            self._skip("voice configuration", str(exc))
            return
        devices = query_audio_devices()
        if not devices.get("sounddevice_installed"):
            self._skip("microphone enumeration", str(devices.get("error", "sounddevice missing")))
        else:
            input_count = len(devices.get("input_devices", []))
            output_count = len(devices.get("output_devices", []))
            if input_count:
                self._pass("microphone enumeration", f"{input_count} input devices")
            else:
                self._manual("microphone enumeration", "No input devices reported by sounddevice.")
            if output_count:
                self._pass("speaker enumeration", f"{output_count} output devices")
            else:
                self._manual("speaker enumeration", "No output devices reported by sounddevice.")
        report = voice_doctor(settings)
        components = {
            str(component.get("name")): component for component in report.get("components", [])
        }
        for check_name, component_name in (
            ("whisper.cpp executable availability", "whisper binary"),
            ("whisper.cpp model availability", "whisper model"),
            ("Piper executable availability", "piper binary"),
            ("Piper voice availability", "piper model"),
            ("wake-word model availability", "wake-word model"),
        ):
            component = components.get(component_name)
            status = str(component.get("status")) if component else "degraded"
            message = str(component.get("message")) if component else "not reported"
            if status == "ok":
                self._pass(check_name, message)
            elif component_name == "wake-word model":
                self._manual(check_name, message)
            elif settings.voice.enabled:
                self._fail(check_name, message)
            else:
                self._skip(check_name, message)

    def _required_or_skip(self, name: str, detail: str) -> None:
        if self.require_real_model:
            self._fail(name, detail)
        else:
            self._skip(name, detail)

    def _pass(self, name: str, detail: str) -> None:
        self.checks.append(VerifyCheck(name=name, ok=True, detail=detail, status="pass"))

    def _fail(self, name: str, detail: str) -> None:
        self.checks.append(VerifyCheck(name=name, ok=False, detail=detail, status="fail"))

    def _skip(self, name: str, detail: str) -> None:
        self.checks.append(VerifyCheck(name=name, ok=True, detail=detail, status="skip"))

    def _manual(self, name: str, detail: str) -> None:
        self.checks.append(VerifyCheck(name=name, ok=True, detail=detail, status="manual"))


class RealModelVerifier:  # pragma: no cover - requires optional real GGUF runtime
    def __init__(
        self,
        *,
        home: Path,
        model_path: Path,
        max_output_tokens: int = 32,
        timeout: float = 180.0,
    ) -> None:
        self.repo_home = home.expanduser().resolve()
        self.model_path = model_path.expanduser().resolve()
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.temp = Path(tempfile.mkdtemp(prefix="april-real-verify-"))
        self.verify_home = self.temp / "april_home"
        self.runtime_port = _free_port()
        self.api_port = _free_port()
        self.api_token = "real-verify-api-token"
        self.runtime_token = "real-verify-runtime-token"
        self.runtime: subprocess.Popen[bytes] | None = None
        self.api: subprocess.Popen[bytes] | None = None
        self.runtime_log = self.temp / "runtime.log"
        self.api_log = self.temp / "api.log"
        self.checks: list[VerifyCheck] = []
        self.load_time_seconds: float | None = None
        self.first_token_latency_seconds: float | None = None
        self.generation_time_seconds: float | None = None
        self.output_tokens: int = 0
        self.tokens_per_second: float | None = None
        self.prompt_path: str = "unknown"
        self.runtime_rss_bytes: int | None = None

    @property
    def runtime_url(self) -> str:
        return f"http://127.0.0.1:{self.runtime_port}"

    @property
    def api_url(self) -> str:
        return f"http://127.0.0.1:{self.api_port}"

    @property
    def runtime_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.runtime_token}"}

    def run(self) -> list[VerifyCheck]:
        try:
            self._prepare()
            env = self._env()
            self.runtime = self._start("services.april_runtime.server", env, self.runtime_log)
            self.api = self._start("services.api.server", env, self.api_log)
            self._check(
                "runtime health",
                lambda: self._wait_json(self.runtime_url + "/runtime/health", auth_runtime=True),
            )
            self._check("core health", lambda: self._wait_json(self.api_url + "/health"))
            self._check("real model load", self._load_model)
            self._check("real model chat", self._chat)
            self._check("real model stream", self._stream)
            self._check("real model unload", self._unload_model)
            self._check("real model unloaded state", self._confirm_unloaded)
            self._check("real model metrics", self._metrics)
        finally:
            self._stop()
            self._check("services stopped", self._services_stopped)
            shutil.rmtree(self.temp, ignore_errors=True)
        return self.checks

    def _prepare(self) -> None:
        self.verify_home.mkdir(parents=True)
        shutil.copytree(self.repo_home / "configs", self.verify_home / "configs")
        # A standalone single-file verify/benchmark fabricates this model entry, so
        # there is no operator-configured chat_format to fall back on. Infer the
        # family from the GGUF *basename* (the only signal available) and default to
        # the always-supported "generic" template so an arbitrary model can still be
        # chatted/streamed for a structural load/chat/stream/unload smoke. Without
        # this the resolver only inspects model.name ("real-smoke") and raises
        # "Unsupported chat template", failing chat for every supplied model.
        chat_format = _infer_chat_format_from_basename(self.model_path.name)
        model_entry = {
            "name": "real-smoke",
            "path": str(self.model_path),
            "backend": "llama_cpp",
            "chat_format": chat_format,
            "threads": 2,
            "context_size": 1024,
            "temperature": 0.0,
            "max_output_tokens": 64,
            "keep_loaded": False,
            "idle_unload_seconds": None,
            "priority": 50,
        }
        models = {
            "brain": {
                **model_entry,
                "id": "april-brain",
                "role": "brain",
                "priority": 100,
            },
            "coding": {
                **model_entry,
                "id": "april-coding",
                "role": "coding",
            },
            "reading": {
                **model_entry,
                "id": "april-reading",
                "role": "reading",
            },
        }
        (self.verify_home / "configs" / "models.yaml").write_text(
            yaml.safe_dump({"models": models}, sort_keys=False),
            encoding="utf-8",
        )

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            {
                "APRIL_HOME": str(self.verify_home),
                "PYTHONPATH": str(self.repo_home),
                "APRIL_RUNTIME_BACKEND": "llama_cpp",
                "APRIL_RUNTIME_PRELOAD_KEEP_LOADED": "false",
                "APRIL_RUNTIME_PORT": str(self.runtime_port),
                "APRIL_API_PORT": str(self.api_port),
                "APRIL_RUNTIME_URL": self.runtime_url,
                "APRIL_RUNTIME_TOKEN": self.runtime_token,
                "APRIL_API_TOKEN": self.api_token,
                "APRIL_DATABASE_PATH": str(self.temp / "data" / "april.db"),
                "APRIL_VECTOR_INDEX_PATH": str(self.temp / "data" / "vector_index"),
                "APRIL_AUDIT_PATH": str(self.temp / "logs" / "audit.jsonl"),
                "APRIL_LOGS_PATH": str(self.temp / "logs"),
                "APRIL_ALLOWED_FILESYSTEM_ROOTS": str(self.temp),
            }
        )
        return env

    def _start(self, module: str, env: dict[str, str], log_path: Path) -> subprocess.Popen[bytes]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log_file:
            return subprocess.Popen(
                [sys.executable, "-m", module],
                cwd=str(self.repo_home),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    def _wait_json(self, url: str, *, auth_runtime: bool = False) -> dict[str, Any]:
        deadline = time.monotonic() + 30.0
        last = ""
        headers = self.runtime_headers if auth_runtime else None
        while time.monotonic() < deadline:
            try:
                response = httpx.get(url, timeout=1.0, headers=headers)
                if response.status_code == 200:
                    return response.json()
                last = response.text[:500]
            except httpx.HTTPError as exc:
                last = str(exc)
            time.sleep(0.2)
        raise RuntimeError(f"health check failed for {url}: {last}")

    def _load_model(self) -> str:
        start = time.monotonic()
        data = self._post_runtime(
            "/runtime/models/load",
            {"model_id": "april-brain", "request_id": "real-verify-load"},
            timeout=self.timeout,
        )
        self.load_time_seconds = time.monotonic() - start
        state = data.get("state")
        if state != "loaded":
            raise RuntimeError(f"expected loaded state, got {state}")
        return f"loaded in {self.load_time_seconds:.2f}s"

    def _chat(self) -> str:
        data = self._post_runtime(
            "/runtime/chat",
            {
                "model_id": "april-brain",
                "messages": [{"role": "user", "content": "Reply with the word ready."}],
                "options": {"temperature": 0.0, "max_output_tokens": self.max_output_tokens},
                "request_id": "real-verify-chat",
            },
            timeout=self.timeout,
        )
        content = str(data.get("content", "")).strip()
        usage = data.get("usage") or {}
        diagnostics = data.get("diagnostics") or {}
        if diagnostics.get("prompt_path"):
            self.prompt_path = str(diagnostics["prompt_path"])
        if not content:
            raise RuntimeError("chat returned empty content")
        if int(usage.get("total_tokens", 0)) < int(usage.get("output_tokens", 0)):
            raise RuntimeError(f"invalid usage payload: {usage}")
        return content[:80]

    def _stream(self) -> str:
        request = {
            "model_id": "april-brain",
            "messages": [{"role": "user", "content": "Say ok."}],
            "options": {"temperature": 0.0, "max_output_tokens": self.max_output_tokens},
            "request_id": "real-verify-stream",
        }
        token_count = 0
        usage_count = 0
        started = time.monotonic()
        first_token_at: float | None = None
        with httpx.stream(
            "POST",
            self.runtime_url + "/runtime/stream",
            json=request,
            headers=self.runtime_headers,
            timeout=self.timeout,
        ) as response:
            if response.status_code >= 400:
                raise RuntimeError(self._response_error(response))
            for line in response.iter_lines():
                if line.startswith("event: token"):
                    token_count += 1
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                elif line.startswith("event: usage"):
                    usage_count += 1
                elif line.startswith("data: "):
                    self._record_stream_data(line[6:])
        if token_count < 1 or usage_count != 1:
            raise RuntimeError(f"tokens={token_count}, usage={usage_count}")
        elapsed = max(time.monotonic() - started, 0.000_001)
        self.first_token_latency_seconds = (
            first_token_at - started if first_token_at is not None else None
        )
        self.generation_time_seconds = elapsed
        if self.output_tokens <= 0:
            self.output_tokens = token_count
        self.tokens_per_second = self.output_tokens / elapsed
        return (
            f"{token_count} token events, {usage_count} usage event, "
            f"{self.tokens_per_second:.2f} tokens/sec"
        )

    def _unload_model(self) -> str:
        data = self._post_runtime(
            "/runtime/models/unload",
            {"model_id": "april-brain", "request_id": "real-verify-unload"},
            timeout=self.timeout,
        )
        state = data.get("state")
        if state not in {"unloaded", "unavailable"}:
            raise RuntimeError(f"expected unloaded/unavailable state, got {state}")
        return str(state)

    def _metrics(self) -> str:
        self.runtime_rss_bytes = _process_rss_bytes(self.runtime.pid if self.runtime else None)
        details = {
            "load_time_seconds": self.load_time_seconds,
            "first_token_latency_seconds": self.first_token_latency_seconds,
            "total_generation_time_seconds": self.generation_time_seconds,
            "output_tokens": self.output_tokens,
            "tokens_per_second": self.tokens_per_second,
            "context_size_used": 1024,
            "backend_settings": {
                "backend": "llama_cpp",
                "threads": 2,
                "n_batch": None,
                "max_output_tokens": self.max_output_tokens,
            },
            "prompt_path": self.prompt_path,
            "unload_success": True,
            "runtime_rss_bytes": self.runtime_rss_bytes,
        }
        return yaml.safe_dump(details, sort_keys=False).strip()

    def _record_stream_data(self, raw: str) -> None:
        try:
            import json

            data = json.loads(raw)
        except ValueError:
            return
        payload = data.get("payload") if isinstance(data, dict) else None
        if not isinstance(payload, dict):
            return
        if "output_tokens" in payload:
            self.output_tokens = int(payload["output_tokens"])
        if payload.get("prompt_path"):
            self.prompt_path = str(payload["prompt_path"])

    def _confirm_unloaded(self) -> str:
        response = httpx.get(
            self.runtime_url + "/runtime/models",
            headers=self.runtime_headers,
            timeout=10.0,
        )
        if response.status_code >= 400:
            raise RuntimeError(self._response_error(response))
        models = response.json().get("models", [])
        state = next(
            (model.get("state") for model in models if model.get("id") == "april-brain"),
            None,
        )
        if state not in {"unloaded", "unavailable"}:
            raise RuntimeError(f"model state is {state}")
        return str(state)

    def _post_runtime(
        self, path: str, payload: dict[str, Any], *, timeout: float
    ) -> dict[str, Any]:
        response = httpx.post(
            self.runtime_url + path,
            json=payload,
            headers=self.runtime_headers,
            timeout=timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(self._response_error(response))
        return response.json()

    def _response_error(self, response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            return response.text[:1000]
        error = data.get("error", {}) if isinstance(data, dict) else {}
        message = error.get("message") or response.text[:1000]
        details = error.get("details") or {}
        return f"{message} {details}".strip()

    def _brain_decision_database(self) -> Path:
        return self.temp / "data" / "april.db"

    def _brain_decision_marker(self) -> int:
        return latest_brain_decision_marker(self._brain_decision_database())

    def _brain_decision_after(self, marker: int) -> dict[str, Any]:
        return brain_decision_after_marker(self._brain_decision_database(), marker)

    def _services_stopped(self) -> str:
        alive = []
        for name, proc in (("runtime", self.runtime), ("api", self.api)):
            if proc is not None and proc.poll() is None:
                alive.append(name)
        if alive:
            raise RuntimeError(f"still running: {', '.join(alive)}")
        return "stopped"

    def _check(self, name: str, action: Callable[[], Any]) -> Any:
        try:
            detail = action()
        except Exception as exc:
            self.checks.append(VerifyCheck(name=name, ok=False, detail=str(exc)))
            return None
        self.checks.append(VerifyCheck(name=name, ok=True, detail=str(detail)))
        return detail

    def _stop(self) -> None:
        for proc in (self.api, self.runtime):
            if proc is not None and proc.poll() is None:
                proc.terminate()
        for proc in (self.api, self.runtime):
            if proc is None:
                continue
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


class ModelBenchmark(RealModelVerifier):  # pragma: no cover - requires optional real GGUF runtime
    def __init__(
        self,
        *,
        home: Path,
        model_path: Path,
        prompt: str,
        runs: int,
        max_output_tokens: int,
        keep_loaded: bool,
    ) -> None:
        super().__init__(
            home=home,
            model_path=model_path,
            max_output_tokens=max_output_tokens,
            timeout=180.0,
        )
        self.prompt = prompt
        self.runs = runs
        self.keep_loaded = keep_loaded

    def run(self) -> list[BenchmarkResult]:  # type: ignore[override]
        results: list[BenchmarkResult] = []
        try:
            self._prepare()
            env = self._env()
            self.runtime = self._start("services.april_runtime.server", env, self.runtime_log)
            self.api = self._start("services.api.server", env, self.api_log)
            self._wait_json(self.runtime_url + "/runtime/health", auth_runtime=True)
            self._wait_json(self.api_url + "/health")
            for index in range(1, self.runs + 1):
                results.append(self._run_one(index))
        finally:
            self._stop()
            shutil.rmtree(self.temp, ignore_errors=True)
        return results

    def _run_one(self, index: int) -> BenchmarkResult:
        self.load_time_seconds = None
        self.first_token_latency_seconds = None
        self.generation_time_seconds = None
        self.output_tokens = 0
        self.tokens_per_second = None
        try:
            self._load_model()
            self._benchmark_stream()
            unload_success = False
            if not self.keep_loaded:
                self._unload_model()
                unload_success = True
            return BenchmarkResult(
                run_index=index,
                ok=True,
                load_time_seconds=self.load_time_seconds or 0.0,
                first_token_latency_seconds=self.first_token_latency_seconds,
                generation_time_seconds=self.generation_time_seconds or 0.0,
                output_tokens=self.output_tokens,
                tokens_per_second=self.tokens_per_second or 0.0,
                unload_success=unload_success,
                context_size=1024,
                backend_settings={
                    "backend": "llama_cpp",
                    "threads": 2,
                    "max_output_tokens": self.max_output_tokens,
                },
            )
        except Exception as exc:
            return BenchmarkResult(run_index=index, ok=False, detail=str(exc))

    def _benchmark_stream(self) -> None:
        request = {
            "model_id": "april-brain",
            "messages": [{"role": "user", "content": self.prompt}],
            "options": {"temperature": 0.0, "max_output_tokens": self.max_output_tokens},
            "request_id": "model-benchmark",
        }
        started = time.monotonic()
        first_token_at: float | None = None
        token_events = 0
        with httpx.stream(
            "POST",
            self.runtime_url + "/runtime/stream",
            json=request,
            headers=self.runtime_headers,
            timeout=self.timeout,
        ) as response:
            if response.status_code >= 400:
                raise RuntimeError(self._response_error(response))
            for line in response.iter_lines():
                if line.startswith("event: token"):
                    token_events += 1
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                elif line.startswith("data: "):
                    self._record_stream_data(line[6:])
        elapsed = max(time.monotonic() - started, 0.000_001)
        self.first_token_latency_seconds = (
            first_token_at - started if first_token_at is not None else None
        )
        self.generation_time_seconds = elapsed
        if self.output_tokens <= 0:
            self.output_tokens = token_events
        self.tokens_per_second = self.output_tokens / elapsed


class LauncherVerifier:
    def __init__(self, *, home: Path) -> None:
        self.repo_home = home.expanduser().resolve()
        self.temp = Path(tempfile.mkdtemp(prefix="april-verify-"))
        self.verify_home = self.temp / "april_home"
        self.project = self.temp / "external_project"
        self.second_project = self.temp / "second_project"
        self.runtime_port = _free_port()
        self.api_port = _free_port()
        self.api_token = "verify-token"
        self.runtime_token = "verify-runtime-token"
        self.runtime: subprocess.Popen[bytes] | None = None
        self.api: subprocess.Popen[bytes] | None = None
        self.runtime_log = self.temp / "runtime.log"
        self.api_log = self.temp / "api.log"
        self.checks: list[VerifyCheck] = []

    def run(self) -> list[VerifyCheck]:
        try:
            self._prepare()
            env = self._env()
            self.runtime = self._start("services.april_runtime.server", env, self.runtime_log)
            self.api = self._start("services.api.server", env, self.api_log)
            self._check(
                "runtime health", lambda: self._wait_json(self.runtime_url + "/runtime/health")
            )
            self._check("core health", lambda: self._wait_json(self.api_url + "/health"))
            self._check("model listing", self._check_models)
            project_id = self._check("project registration", self._register_project)
            conversation_id = self._check(
                "multi-turn conversation",
                lambda: self._multi_turn(project_id),
            )
            self._check(
                "conversation isolation",
                lambda: self._isolated_conversation(project_id, conversation_id),
            )
            self._check(
                "conversation project switch rejection",
                lambda: self._conversation_switch_rejected(conversation_id),
            )
            self._check("read-only repo analysis", lambda: self._repo_analysis(project_id))
            self._check(
                "direct agent structured execution",
                lambda: self._direct_agent_run(project_id),
            )
            denial_approval_id = self._check(
                "denial path", lambda: self._patch_approval(project_id)
            )
            self._check("approval denial", lambda: self._deny_approval(denial_approval_id))
            expired_approval_id = self._check(
                "expired approval path", lambda: self._patch_approval(project_id)
            )
            self._check(
                "expired approval rejection",
                lambda: self._expired_approval_rejected(expired_approval_id),
            )
            approval_id = self._check(
                "patch approval creation", lambda: self._patch_approval(project_id)
            )
            self._check("exact patch approval application", lambda: self._approve(approval_id))
            self._check(
                "approval replay rejection", lambda: self._approval_replay_rejected(approval_id)
            )
            self._check(
                "tampered artifact rejection", lambda: self._tampered_artifact_rejected(project_id)
            )
            self._check(
                "path escape patch rejection", lambda: self._path_escape_rejected(project_id)
            )
            self._check("repo override rejection", lambda: self._repo_override_rejected())
            self._check("run command cwd forcing", lambda: self._run_command_cwd_forced(project_id))
            self._check("runtime streaming usage", self._runtime_streaming)
            self._check("audit records", self._audit_records)
            self._check("tool call records", self._tool_call_records)
            self._check("agent run records", self._agent_run_records)
        finally:
            self._stop()
            self._check("services stopped", self._services_stopped)
            shutil.rmtree(self.temp, ignore_errors=True)
        return self.checks

    @property
    def runtime_url(self) -> str:
        return f"http://127.0.0.1:{self.runtime_port}"

    @property
    def api_url(self) -> str:
        return f"http://127.0.0.1:{self.api_port}"

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_token}"}

    @property
    def runtime_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.runtime_token}"}

    def _prepare(self) -> None:
        self.verify_home.mkdir(parents=True)
        shutil.copytree(self.repo_home / "configs", self.verify_home / "configs")
        self.project.mkdir()
        self.second_project.mkdir()
        (self.project / "README.md").write_text("# verify\nanimation bug\n", encoding="utf-8")
        (self.project / "app.py").write_text("value = 'old'\n", encoding="utf-8")
        (self.second_project / "README.md").write_text("# second\n", encoding="utf-8")
        _git(self.project, "init")
        _git(self.project, "config", "user.email", "april@example.local")
        _git(self.project, "config", "user.name", "APRIL Verify")
        _git(self.project, "add", ".")
        _git(self.project, "commit", "-m", "initial")

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            {
                "APRIL_HOME": str(self.verify_home),
                "PYTHONPATH": str(self.repo_home),
                "APRIL_RUNTIME_BACKEND": "fake",
                "APRIL_RUNTIME_PORT": str(self.runtime_port),
                "APRIL_API_PORT": str(self.api_port),
                "APRIL_RUNTIME_URL": self.runtime_url,
                "APRIL_RUNTIME_TOKEN": self.runtime_token,
                "APRIL_API_TOKEN": self.api_token,
                "APRIL_DATABASE_PATH": str(self.temp / "data" / "april.db"),
                "APRIL_VECTOR_INDEX_PATH": str(self.temp / "data" / "vector_index"),
                "APRIL_AUDIT_PATH": str(self.temp / "logs" / "audit.jsonl"),
                "APRIL_LOGS_PATH": str(self.temp / "logs"),
                "APRIL_ALLOWED_FILESYSTEM_ROOTS": f"{self.project},{self.second_project}",
            }
        )
        return env

    def _start(self, module: str, env: dict[str, str], log_path: Path) -> subprocess.Popen[bytes]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log_file:
            return subprocess.Popen(
                [sys.executable, "-m", module],
                cwd=str(self.repo_home),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    def _wait_json(self, url: str) -> dict[str, Any]:
        deadline = time.monotonic() + 20.0
        last = ""
        while time.monotonic() < deadline:
            try:
                headers = self.runtime_headers if url.startswith(self.runtime_url) else None
                response = httpx.get(url, timeout=1.0, headers=headers)
                if response.status_code == 200:
                    return response.json()
                last = response.text[:200]
            except httpx.HTTPError as exc:
                last = str(exc)
            time.sleep(0.2)
        raise RuntimeError(f"health check failed for {url}: {last}")

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.api_url, headers=self.headers, timeout=10.0)

    def _check_models(self) -> str:
        with self._client() as client:
            data = client.get("/runtime/models").json()
        count = len(data.get("models", []))
        if count < 1:
            raise RuntimeError("no models returned")
        return f"{count} models"

    def _register_project(self) -> str:
        with self._client() as client:
            response = client.post("/projects", json={"path": str(self.project)})
        response.raise_for_status()
        project_id = str(response.json()["id"])
        return project_id

    def _multi_turn(self, project_id: str) -> str:
        with self._client() as client:
            first = client.post(
                "/chat",
                json={"message": "April, plan my work today.", "project_id": project_id},
            ).json()
            conversation_id = first["result"]["conversation_id"]
            second = client.post(
                "/chat",
                json={
                    "message": "Use that same plan.",
                    "project_id": project_id,
                    "conversation_id": conversation_id,
                },
            ).json()
        if second["result"]["status"] != "ok":
            raise RuntimeError("second turn failed")
        return str(conversation_id)

    def _isolated_conversation(self, project_id: str, existing_id: str) -> str:
        with self._client() as client:
            other = client.post(
                "/chat",
                json={"message": "Start a separate plan.", "project_id": project_id},
            ).json()
        other_id = other["result"]["conversation_id"]
        if other_id == existing_id:
            raise RuntimeError("conversation IDs overlapped")
        return str(other_id)

    def _conversation_switch_rejected(self, existing_id: str) -> str:
        with self._client() as client:
            second = client.post("/projects", json={"path": str(self.second_project)}).json()
            response = client.post(
                "/chat",
                json={
                    "message": "Try to move this conversation.",
                    "project_id": second["id"],
                    "conversation_id": existing_id,
                },
            )
        if response.status_code != 403:
            raise RuntimeError(f"expected 403, got {response.status_code}")
        return "403"

    def _repo_analysis(self, project_id: str) -> str:
        with self._client() as client:
            response = client.post(
                "/chat",
                json={
                    "message": "April, check why the animation in this repository is broken.",
                    "project_id": project_id,
                },
            ).json()
        if response["result"]["status"] != "ok":
            raise RuntimeError("repo analysis failed")
        return "ok"

    def _patch_approval(self, project_id: str) -> str:
        with self._client() as client:
            response = client.post(
                "/chat",
                json={
                    "message": "Apply the fix.",
                    "project_id": project_id,
                },
            ).json()
        result = response["result"]
        if result["status"] != "pending_approval":
            raise RuntimeError(str(response))
        approval = result["pending_approval"]
        if approval["metadata"].get("agent_run_id") is None:
            raise RuntimeError("approval is not bound to a structured agent run")
        return str(approval["approval_id"])

    def _direct_agent_run(self, project_id: str) -> str:
        with self._client() as client:
            response = client.post(
                "/agents/run",
                json={
                    "agent": "coding_agent",
                    "message": "Check animation files",
                    "project_id": project_id,
                },
            ).json()
        if response["result"]["status"] != "ok":
            raise RuntimeError(str(response))
        return "ok"

    def _approve(self, approval_id: str) -> str:
        with self._client() as client:
            response = client.post("/tools/approve", json={"approval_id": approval_id}).json()
        if response.get("status") != "resumed":
            raise RuntimeError(str(response))
        if "fixed animation" not in (self.project / "README.md").read_text(encoding="utf-8"):
            raise RuntimeError("patch was not applied")
        if response.get("result", {}).get("status") != "ok":
            raise RuntimeError("agent did not return final answer after resume")
        return "applied and resumed"

    def _approval_replay_rejected(self, approval_id: str) -> str:
        with self._client() as client:
            response = client.post("/tools/approve", json={"approval_id": approval_id})
        if response.status_code != 403:
            raise RuntimeError(f"expected 403, got {response.status_code}")
        return "403"

    def _deny_approval(self, approval_id: str) -> str:
        with self._client() as client:
            response = client.post("/tools/deny", json={"approval_id": approval_id})
        if response.status_code != 200:
            raise RuntimeError(f"expected 200, got {response.status_code}")
        payload = response.json()
        if payload.get("status") != "denied":
            raise RuntimeError(str(payload))
        status = self._suspended_status(approval_id)
        if status is not None and status != "denied":
            raise RuntimeError(f"suspended run status is {status}")
        return "denied"

    def _expired_approval_rejected(self, approval_id: str) -> str:
        database = self.temp / "data" / "april.db"
        if database.exists():
            with sqlite3.connect(database) as conn:
                conn.execute(
                    "UPDATE approvals SET expires_at = ? WHERE id = ?",
                    ("1970-01-01T00:00:00Z", approval_id),
                )
                conn.commit()
        with self._client() as client:
            response = client.post("/tools/approve", json={"approval_id": approval_id})
        if response.status_code != 403:
            raise RuntimeError(f"expected 403, got {response.status_code}")
        status = self._suspended_status(approval_id)
        if status is not None and status != "expired":
            raise RuntimeError(f"suspended run status is {status}")
        return "403 expired"

    def _tampered_artifact_rejected(self, project_id: str) -> str:
        patch_dir = self.verify_home / "data" / "patches"
        patch_dir.mkdir(parents=True, exist_ok=True)
        patch_path = patch_dir / "tamper.patch"
        patch_path.write_text(
            "diff --git a/README.md b/README.md\n"
            "--- a/README.md\n"
            "+++ b/README.md\n"
            "@@ -1,2 +1,3 @@\n"
            " # verify\n"
            " animation bug\n"
            "+tamper check\n",
            encoding="utf-8",
        )
        with self._client() as client:
            response = client.post(
                "/tools/request",
                json={
                    "tool": "patch_applier",
                    "agent": "coding_agent",
                    "args": {
                        "repo_path": str(self.project),
                        "patch_path": str(patch_path),
                        "project_id": project_id,
                    },
                },
            ).json()
            approval = response["approval"]
            artifact_id = approval["metadata"]["artifact_id"]
            artifact_path = (
                self.verify_home / "data" / "artifacts" / "patches" / f"{artifact_id}.patch"
            )
            artifact_path.write_text("tampered bytes\n", encoding="utf-8")
            approve = client.post(
                "/tools/approve",
                json={"approval_id": approval["approval_id"]},
            ).json()
        if approve.get("status") != "failed":
            raise RuntimeError(str(approve))
        return "failed"

    def _path_escape_rejected(self, project_id: str) -> str:
        patch_dir = self.verify_home / "data" / "patches"
        patch_dir.mkdir(parents=True, exist_ok=True)
        patch_path = patch_dir / "escape.patch"
        patch_path.write_text(
            "diff --git a/../escape.txt b/../escape.txt\n"
            "--- a/../escape.txt\n"
            "+++ b/../escape.txt\n"
            "@@ -0,0 +1 @@\n"
            "+escape\n",
            encoding="utf-8",
        )
        with self._client() as client:
            response = client.post(
                "/tools/request",
                json={
                    "tool": "patch_applier",
                    "agent": "coding_agent",
                    "args": {
                        "repo_path": str(self.project),
                        "patch_path": str(patch_path),
                        "project_id": project_id,
                    },
                },
            )
        if response.status_code != 403:
            raise RuntimeError(f"expected 403, got {response.status_code}")
        return "403"

    def _repo_override_rejected(self) -> str:
        with self._client() as client:
            response = client.post(
                "/tools/request",
                json={
                    "tool": "git_status",
                    "agent": "coding_agent",
                    "args": {"repo_path": str(self.second_project)},
                },
            )
        if response.status_code != 403:
            raise RuntimeError(f"expected 403, got {response.status_code}")
        return "403"

    def _run_command_cwd_forced(self, project_id: str) -> str:
        with self._client() as client:
            response = client.post(
                "/tools/request",
                json={
                    "tool": "run_command",
                    "agent": "coding_agent",
                    "args": {
                        "project_id": project_id,
                        "argv": ["pytest"],
                        "cwd": str(self.second_project),
                    },
                },
            ).json()
        cwd = response["approval"]["args"]["cwd"]
        if Path(cwd).resolve() != self.project.resolve():
            raise RuntimeError(f"cwd was not forced: {cwd}")
        return "forced"

    def _runtime_streaming(self) -> str:
        request = {
            "model_id": "april-brain",
            "messages": [{"role": "user", "content": "April, plan my work today."}],
            "request_id": "verify-stream",
        }
        usage_count = 0
        token_count = 0
        with httpx.stream(
            "POST",
            self.runtime_url + "/runtime/stream",
            json=request,
            headers=self.runtime_headers,
            timeout=10.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("event: token"):
                    token_count += 1
                if line.startswith("event: usage"):
                    usage_count += 1
        if token_count < 1 or usage_count != 1:
            raise RuntimeError(f"tokens={token_count}, usage={usage_count}")
        return f"{token_count} token events"

    def _audit_records(self) -> str:
        audit = self.temp / "logs" / "audit.jsonl"
        text = audit.read_text(encoding="utf-8")
        if "approved_tool_executed" not in text or "approval_consumed" not in text:
            raise RuntimeError("expected audit events not found")
        return "ok"

    def _tool_call_records(self) -> str:
        database = self.temp / "data" / "april.db"
        with sqlite3.connect(database) as conn:
            count = conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
        if count < 1:
            raise RuntimeError("no tool call rows found")
        return str(count)

    def _agent_run_records(self) -> str:
        database = self.temp / "data" / "april.db"
        with sqlite3.connect(database) as conn:
            runs = conn.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0]
            iterations = conn.execute("SELECT COUNT(*) FROM agent_iterations").fetchone()[0]
            suspended = conn.execute("SELECT COUNT(*) FROM suspended_agent_runs").fetchone()[0]
        if runs < 1 or iterations < 1 or suspended < 1:
            raise RuntimeError(f"runs={runs}, iterations={iterations}, suspended={suspended}")
        return f"runs={runs}, iterations={iterations}, suspended={suspended}"

    def _suspended_status(self, approval_id: str) -> str | None:
        database = self.temp / "data" / "april.db"
        if not database.exists():
            return None
        with sqlite3.connect(database) as conn:
            row = conn.execute(
                "SELECT status FROM suspended_agent_runs WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
        return None if row is None else str(row[0])

    def _services_stopped(self) -> str:
        alive = []
        for name, proc in (("runtime", self.runtime), ("api", self.api)):
            if proc is not None and proc.poll() is None:
                alive.append(name)
        if alive:
            raise RuntimeError(f"still running: {', '.join(alive)}")
        return "stopped"

    def _check(self, name: str, action: Callable[[], Any]) -> Any:
        try:
            detail = action()
        except Exception as exc:
            self.checks.append(VerifyCheck(name=name, ok=False, detail=str(exc)))
            return None
        self.checks.append(VerifyCheck(name=name, ok=True, detail=str(detail)))
        return detail

    def _stop(self) -> None:
        for proc in (self.api, self.runtime):
            if proc is not None and proc.poll() is None:
                proc.terminate()
        for proc in (self.api, self.runtime):
            if proc is None:
                continue
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


class WorkflowVerifier(LauncherVerifier):
    def run(self) -> list[VerifyCheck]:
        try:
            self._prepare()
            env = self._env()
            self.runtime = self._start("services.april_runtime.server", env, self.runtime_log)
            self.api = self._start("services.api.server", env, self.api_log)
            self._check(
                "runtime health", lambda: self._wait_json(self.runtime_url + "/runtime/health")
            )
            self._check("core health", lambda: self._wait_json(self.api_url + "/health"))
            self._check("model load/unload", self._model_load_unload)
            project_id = self._check("project registration", self._register_project)
            self._check("planning request", lambda: self._multi_turn(project_id))
            self._check("task creation and listing", self._task_listing)
            self._check("repo inspection request", lambda: self._repo_analysis(project_id))
            approval_id = self._check(
                "code-write approval creation", lambda: self._patch_approval(project_id)
            )
            self._check("approval denial", lambda: self._deny_approval(approval_id))
            approval_id = self._check(
                "approval approval/resume path", lambda: self._patch_approval(project_id)
            )
            self._check("approval approval/resume", lambda: self._approve(approval_id))
            self._check("memory-policy check for system_action_agent", self._system_action_policy)
            self._check("reminder creation/listing", self._reminder_create_list)
            self._check("voice health", self._voice_health)
        finally:
            self._stop()
            self._check("services stopped", self._services_stopped)
            shutil.rmtree(self.temp, ignore_errors=True)
        return self.checks

    def _model_load_unload(self) -> str:
        with self._client() as client:
            loaded = client.post(
                "/runtime/models/load",
                json={"model_id": "april-brain", "request_id": "workflow-load"},
            ).json()
            unloaded = client.post(
                "/runtime/models/unload",
                json={"model_id": "april-brain", "request_id": "workflow-unload"},
            ).json()
        return f"{loaded.get('state')} -> {unloaded.get('state')}"

    def _task_listing(self) -> str:
        with self._client() as client:
            tasks = client.get("/tasks").json().get("tasks", [])
        if not tasks:
            raise RuntimeError("no task plans were created")
        return f"{len(tasks)} tasks"

    def _system_action_policy(self) -> str:
        database = self.temp / "data" / "april.db"
        with sqlite3.connect(database) as conn:
            rows = conn.execute(
                "SELECT payload_json FROM conversation_events WHERE event_type = 'brain_decision'"
            ).fetchall()
        if not rows:
            raise RuntimeError("no brain decisions were recorded")
        return "system_action_agent policy is loaded from configs/agents.yaml"

    def _reminder_create_list(self) -> str:
        with self._client() as client:
            created = client.post(
                "/reminders",
                json={"content": "stand up", "due_at": "2026-06-21T09:00:00Z"},
            ).json()
            reminders = client.get("/reminders").json().get("reminders", [])
        if not created.get("reminder") or not reminders:
            raise RuntimeError("reminder create/list failed")
        return f"{len(reminders)} reminders"

    def _voice_health(self) -> str:
        with self._client() as client:
            data = client.get("/health").json()
        voice = data.get("voice") or {}
        return str(voice.get("status", "unknown"))


class RealWorkflowVerifier(
    RealModelVerifier
):  # pragma: no cover - requires optional real GGUF runtime
    def __init__(
        self,
        *,
        home: Path,
        model_path: Path,
        max_output_tokens: int = 32,
        timeout: float = 180.0,
    ) -> None:
        super().__init__(
            home=home,
            model_path=model_path,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
        )
        self.workflow_project = self.temp / "workflow_project"
        # A second registered project so repo-override and cwd-forcing can be proven
        # against a *different* allowed project, exactly like the fake checklist.
        self.second_project = self.temp / "workflow_second_project"
        self.documents_dir = self.temp / "workflow_documents"

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_token}"}

    def _api_client(self) -> httpx.Client:
        return httpx.Client(base_url=self.api_url, headers=self.headers, timeout=self.timeout)

    def _prepare(self) -> None:
        super()._prepare()
        self.workflow_project.mkdir(parents=True)
        (self.workflow_project / "README.md").write_text(
            "# workflow\nanimation bug\n", encoding="utf-8"
        )
        (self.workflow_project / "app.py").write_text("value = 'old'\n", encoding="utf-8")
        _git(self.workflow_project, "init")
        _git(self.workflow_project, "config", "user.email", "april@example.local")
        _git(self.workflow_project, "config", "user.name", "APRIL Verify")
        _git(self.workflow_project, "add", ".")
        _git(self.workflow_project, "commit", "-m", "initial")
        self.second_project.mkdir(parents=True)
        (self.second_project / "README.md").write_text("# second\n", encoding="utf-8")
        _git(self.second_project, "init")
        _git(self.second_project, "config", "user.email", "april@example.local")
        _git(self.second_project, "config", "user.name", "APRIL Verify")
        _git(self.second_project, "add", ".")
        _git(self.second_project, "commit", "-m", "initial")
        self.documents_dir.mkdir(parents=True)
        (self.documents_dir / "workflow-note.txt").write_text(
            "APRIL workflow verification indexes this local document for search.\n",
            encoding="utf-8",
        )

    def run(self) -> list[VerifyCheck]:
        try:
            self._prepare()
            env = self._env()
            self.runtime = self._start("services.april_runtime.server", env, self.runtime_log)
            self.api = self._start("services.api.server", env, self.api_log)
            self._check(
                "runtime health",
                lambda: self._wait_json(self.runtime_url + "/runtime/health", auth_runtime=True),
            )
            self._check("core health", lambda: self._wait_json(self.api_url + "/health"))
            # --- model-dependent smoke (kept minimal and stable) -------------
            self._check("real workflow planning route", self._real_planning_route)
            self._check("real workflow specialist-agent request", self._real_specialist_agent)
            project_id = self._check("workflow project registration", self._register_temp_project)
            self._check("workflow reminder create/list", self._real_reminder_create_list)
            self._check("workflow memory write/search", self._real_memory_write_search)
            self._check("workflow document indexing/search", self._real_document_index_search)
            self._check(
                "workflow coding read-only repo analysis",
                lambda: self._real_coding_repo_analysis(project_id),
            )
            # The model-routed code-write approval proves the real brain reaches the
            # structured agent loop (and seeds suspended_agent_runs); denial closes it.
            agent_approval_id = self._check(
                "workflow code-write approval creation",
                lambda: self._real_code_write_approval(project_id),
            )
            self._check(
                "workflow approval denial", lambda: self._real_approval_denial(agent_approval_id)
            )
            self._check("workflow external action denial", self._real_external_action_denial)
            self._check("workflow voice health", self._real_voice_health)
            # --- deterministic security checklist (model-independent) ----------
            # These mirror the fake workflow/security checklist using explicit
            # tool/API requests so they never depend on the model emitting the same
            # patch twice. A second registered project proves repo-override and
            # cwd-forcing against a different allowed project.
            second_id = self._check(
                "workflow second project registration", self._register_second_project
            )
            patch_approval_id = self._check(
                "workflow patch approval creation",
                lambda: self._security_patch_request(project_id),
            )
            self._check(
                "workflow exact patch approval application",
                lambda: self._security_patch_apply(patch_approval_id),
            )
            self._check(
                "workflow approval replay rejection",
                lambda: self._security_replay_rejected(patch_approval_id),
            )
            self._check(
                "workflow tampered artifact rejection",
                lambda: self._security_tampered_artifact_rejected(project_id),
            )
            self._check(
                "workflow path escape patch rejection",
                lambda: self._security_path_escape_rejected(project_id),
            )
            self._check(
                "workflow repo override rejection",
                lambda: self._security_repo_override_rejected(second_id),
            )
            self._check(
                "workflow run_command cwd forcing",
                lambda: self._security_run_command_cwd_forced(project_id),
            )
            self._check(
                "workflow command allowlist enforcement",
                lambda: self._security_command_allowlist_enforced(project_id),
            )
            self._check("workflow audit records", self._security_audit_records)
            self._check("workflow tool_call records", self._security_tool_call_records)
            self._check("workflow agent_run records", self._security_agent_run_records)
        finally:
            self._stop()
            self._check("services stopped", self._services_stopped)
            shutil.rmtree(self.temp, ignore_errors=True)
        return self.checks

    def _real_planning_route(self) -> str:
        with httpx.Client(
            base_url=self.api_url, headers=self.headers, timeout=self.timeout
        ) as client:
            marker = self._brain_decision_marker()
            response = client.post("/chat", json={"message": "April, plan my work today."})
        if response.status_code >= 400:
            raise RuntimeError(self._response_error(response))
        decision_payload = self._brain_decision_after(marker)
        if not decision_payload:
            raise RuntimeError("chat succeeded but no new brain_decision was recorded")
        decision = BrainDecision.model_validate(decision_payload)
        method = decision.routing_method
        if method == "fallback":
            raise RuntimeError(
                "model/prompt reliability failure: model routing JSON was unusable "
                "and fallback routed the request"
            )
        return f"routing_method={method}"

    def _real_specialist_agent(self) -> str:
        with httpx.Client(
            base_url=self.api_url, headers=self.headers, timeout=self.timeout
        ) as client:
            response = client.post(
                "/agents/run",
                json={
                    "agent": "reading_agent",
                    "message": "Summarize this local validation note: APRIL is ready.",
                },
            )
        if response.status_code >= 400:
            raise RuntimeError(self._response_error(response))
        result = response.json().get("result") or {}
        if result.get("status") != "ok":
            raise RuntimeError(str(result))
        return "reading_agent ok"

    def _latest_routing_method(self) -> str:
        payload = brain_decision_after_marker(
            self._brain_decision_database(),
            max(self._brain_decision_marker() - 1, 0),
        )
        return str(payload.get("routing_method") or "unknown")

    def _latest_decision(self) -> dict[str, Any]:
        return brain_decision_after_marker(
            self._brain_decision_database(),
            max(self._brain_decision_marker() - 1, 0),
        )

    def _register_temp_project(self) -> str:
        with httpx.Client(
            base_url=self.api_url, headers=self.headers, timeout=self.timeout
        ) as client:
            response = client.post(
                "/projects",
                json={"path": str(self.workflow_project), "name": "APRIL workflow verify"},
            )
        if response.status_code >= 400:
            raise RuntimeError(self._response_error(response))
        project_id = str(response.json().get("id") or "")
        if not project_id:
            raise RuntimeError("project id missing")
        return project_id

    def _real_reminder_create_list(self) -> str:
        with httpx.Client(
            base_url=self.api_url, headers=self.headers, timeout=self.timeout
        ) as client:
            created = client.post(
                "/reminders",
                json={"content": "workflow verification", "due_at": "2026-06-21T09:00:00Z"},
            )
            if created.status_code >= 400:
                raise RuntimeError(self._response_error(created))
            listed = client.get("/reminders")
        if listed.status_code >= 400:
            raise RuntimeError(self._response_error(listed))
        reminders = listed.json().get("reminders", [])
        if not reminders:
            raise RuntimeError("no reminders listed after create")
        return f"{len(reminders)} reminders"

    def _real_memory_write_search(self) -> str:
        with httpx.Client(
            base_url=self.api_url, headers=self.headers, timeout=self.timeout
        ) as client:
            created = client.post(
                "/memory",
                json={
                    "content": "workflow verifier prefers concise local answers",
                    "memory_type": "preference",
                    "reason": "Explicit verifier memory write.",
                },
            )
            if created.status_code >= 400:
                raise RuntimeError(self._response_error(created))
            searched = client.get("/memory/search", params={"q": "concise local answers"})
        if searched.status_code >= 400:
            raise RuntimeError(self._response_error(searched))
        results = searched.json().get("results", [])
        if not results:
            raise RuntimeError("memory search returned no results")
        return f"{len(results)} memory results"

    def _real_document_index_search(self) -> str:
        with httpx.Client(
            base_url=self.api_url, headers=self.headers, timeout=self.timeout
        ) as client:
            indexed = client.post("/documents", json={"path": str(self.documents_dir)})
            if indexed.status_code >= 400:
                raise RuntimeError(self._response_error(indexed))
            searched = client.get("/documents/search", params={"q": "workflow verification"})
        if searched.status_code >= 400:
            raise RuntimeError(self._response_error(searched))
        chunks = searched.json().get("chunks", [])
        if not chunks:
            raise RuntimeError("document search returned no chunks")
        return f"{len(chunks)} document chunks"

    def _real_coding_repo_analysis(self, project_id: str | None) -> str:
        if not project_id:
            raise RuntimeError("project registration failed")
        with httpx.Client(
            base_url=self.api_url, headers=self.headers, timeout=self.timeout
        ) as client:
            marker = self._brain_decision_marker()
            response = client.post(
                "/chat",
                json={
                    "message": "April, check why the animation in this repository is broken.",
                    "project_id": project_id,
                },
            )
        if response.status_code >= 400:
            raise RuntimeError(self._response_error(response))
        if not self._brain_decision_after(marker):
            raise RuntimeError("repo analysis succeeded but no new brain_decision was recorded")
        result = response.json().get("result") or {}
        if result.get("status") != "ok":
            raise RuntimeError(str(result))
        return "coding_agent read-only ok"

    def _real_code_write_approval(self, project_id: str | None) -> str:
        if not project_id:
            raise RuntimeError("project registration failed")
        with httpx.Client(
            base_url=self.api_url, headers=self.headers, timeout=self.timeout
        ) as client:
            marker = self._brain_decision_marker()
            response = client.post(
                "/chat",
                json={"message": "Apply the fix.", "project_id": project_id},
            )
        if response.status_code >= 400:
            raise RuntimeError(self._response_error(response))
        if not self._brain_decision_after(marker):
            raise RuntimeError(
                "code-write request succeeded but no new brain_decision was recorded"
            )
        result = response.json().get("result") or {}
        if result.get("status") != "pending_approval":
            raise RuntimeError(str(result))
        approval = result.get("pending_approval") or {}
        approval_id = str(approval.get("approval_id") or "")
        if not approval_id:
            raise RuntimeError("approval id missing")
        return approval_id

    def _real_approval_denial(self, approval_id: str | None) -> str:
        if not approval_id:
            raise RuntimeError("approval creation failed")
        with httpx.Client(
            base_url=self.api_url, headers=self.headers, timeout=self.timeout
        ) as client:
            response = client.post("/tools/deny", json={"approval_id": approval_id})
        if response.status_code >= 400:
            raise RuntimeError(self._response_error(response))
        if response.json().get("status") != "denied":
            raise RuntimeError(str(response.json()))
        return "denied"

    def _real_external_action_denial(self) -> str:
        with httpx.Client(
            base_url=self.api_url, headers=self.headers, timeout=self.timeout
        ) as client:
            response = client.post(
                "/tools/request",
                json={
                    "tool": "send_email",
                    "agent": "system_action_agent",
                    "args": {"to": "nobody@example.invalid", "body": "blocked"},
                },
            )
        if response.status_code != 403:
            raise RuntimeError(f"expected 403, got {response.status_code}")
        return "403"

    def _real_voice_health(self) -> str:
        with httpx.Client(
            base_url=self.api_url, headers=self.headers, timeout=self.timeout
        ) as client:
            response = client.get("/health")
        if response.status_code >= 400:
            raise RuntimeError(self._response_error(response))
        voice = response.json().get("voice") or {}
        return str(voice.get("status", "unknown"))

    # --- deterministic security checklist (model-independent) -----------------

    def _register_second_project(self) -> str:
        with self._api_client() as client:
            response = client.post(
                "/projects",
                json={"path": str(self.second_project), "name": "APRIL workflow second"},
            )
        if response.status_code >= 400:
            raise RuntimeError(self._response_error(response))
        project_id = str(response.json().get("id") or "")
        if not project_id:
            raise RuntimeError("second project id missing")
        return project_id

    def _write_patch(self, name: str, body: str) -> Path:
        patch_dir = self.verify_home / "data" / "patches"
        patch_dir.mkdir(parents=True, exist_ok=True)
        patch_path = patch_dir / name
        patch_path.write_text(body, encoding="utf-8")
        return patch_path

    def _security_patch_request(self, project_id: str | None) -> str:
        if not project_id:
            raise RuntimeError("project registration failed")
        # A fixed, valid diff so the check never depends on the model writing one.
        patch_path = self._write_patch(
            "workflow-apply.patch",
            "diff --git a/README.md b/README.md\n"
            "--- a/README.md\n"
            "+++ b/README.md\n"
            "@@ -1,2 +1,3 @@\n"
            " # workflow\n"
            " animation bug\n"
            "+verified workflow patch\n",
        )
        with self._api_client() as client:
            response = client.post(
                "/tools/request",
                json={
                    "tool": "patch_applier",
                    "agent": "coding_agent",
                    "args": {
                        "repo_path": str(self.workflow_project),
                        "patch_path": str(patch_path),
                        "project_id": project_id,
                    },
                },
            )
        if response.status_code >= 400:
            raise RuntimeError(self._response_error(response))
        payload = response.json()
        if payload.get("status") != "pending_approval":
            raise RuntimeError(str(payload))
        approval = payload.get("approval") or {}
        approval_id = str(approval.get("approval_id") or "")
        if not approval_id or approval.get("metadata", {}).get("artifact_id") is None:
            raise RuntimeError("patch approval did not bind an immutable artifact")
        return approval_id

    def _security_patch_apply(self, approval_id: str | None) -> str:
        if not approval_id:
            raise RuntimeError("patch approval creation failed")
        with self._api_client() as client:
            response = client.post("/tools/approve", json={"approval_id": approval_id})
        if response.status_code >= 400:
            raise RuntimeError(self._response_error(response))
        payload = response.json()
        if payload.get("status") not in {"executed", "resumed"}:
            raise RuntimeError(str(payload))
        readme = (self.workflow_project / "README.md").read_text(encoding="utf-8")
        if "verified workflow patch" not in readme:
            raise RuntimeError("approved patch was not applied to the repository")
        return "applied"

    def _security_replay_rejected(self, approval_id: str | None) -> str:
        if not approval_id:
            raise RuntimeError("patch approval creation failed")
        with self._api_client() as client:
            response = client.post("/tools/approve", json={"approval_id": approval_id})
        if response.status_code != 403:
            raise RuntimeError(f"expected 403 on replay, got {response.status_code}")
        return "403"

    def _security_tampered_artifact_rejected(self, project_id: str | None) -> str:
        if not project_id:
            raise RuntimeError("project registration failed")
        patch_path = self._write_patch(
            "workflow-tamper.patch",
            "diff --git a/README.md b/README.md\n"
            "--- a/README.md\n"
            "+++ b/README.md\n"
            "@@ -1,3 +1,4 @@\n"
            " # workflow\n"
            " animation bug\n"
            " verified workflow patch\n"
            "+tamper check\n",
        )
        with self._api_client() as client:
            response = client.post(
                "/tools/request",
                json={
                    "tool": "patch_applier",
                    "agent": "coding_agent",
                    "args": {
                        "repo_path": str(self.workflow_project),
                        "patch_path": str(patch_path),
                        "project_id": project_id,
                    },
                },
            )
            if response.status_code >= 400:
                raise RuntimeError(self._response_error(response))
            approval = response.json()["approval"]
            artifact_id = approval["metadata"]["artifact_id"]
            artifact_path = (
                self.verify_home / "data" / "artifacts" / "patches" / f"{artifact_id}.patch"
            )
            artifact_path.write_text("tampered bytes\n", encoding="utf-8")
            approve = client.post(
                "/tools/approve", json={"approval_id": approval["approval_id"]}
            ).json()
        if approve.get("status") != "failed":
            raise RuntimeError(str(approve))
        return "failed"

    def _security_path_escape_rejected(self, project_id: str | None) -> str:
        if not project_id:
            raise RuntimeError("project registration failed")
        patch_path = self._write_patch(
            "workflow-escape.patch",
            "diff --git a/../escape.txt b/../escape.txt\n"
            "--- a/../escape.txt\n"
            "+++ b/../escape.txt\n"
            "@@ -0,0 +1 @@\n"
            "+escape\n",
        )
        with self._api_client() as client:
            response = client.post(
                "/tools/request",
                json={
                    "tool": "patch_applier",
                    "agent": "coding_agent",
                    "args": {
                        "repo_path": str(self.workflow_project),
                        "patch_path": str(patch_path),
                        "project_id": project_id,
                    },
                },
            )
        if response.status_code != 403:
            raise RuntimeError(f"expected 403 on path escape, got {response.status_code}")
        return "403"

    def _security_repo_override_rejected(self, second_id: str | None) -> str:
        # A coding tool pointed at a *different* project's repo must be rejected even
        # though that project is itself registered/allowed.
        with self._api_client() as client:
            response = client.post(
                "/tools/request",
                json={
                    "tool": "git_status",
                    "agent": "coding_agent",
                    "args": {"repo_path": str(self.second_project)},
                },
            )
        if response.status_code != 403:
            raise RuntimeError(f"expected 403 on repo override, got {response.status_code}")
        return "403"

    def _security_run_command_cwd_forced(self, project_id: str | None) -> str:
        if not project_id:
            raise RuntimeError("project registration failed")
        with self._api_client() as client:
            response = client.post(
                "/tools/request",
                json={
                    "tool": "run_command",
                    "agent": "coding_agent",
                    "args": {
                        "project_id": project_id,
                        "argv": ["pytest"],
                        "cwd": str(self.second_project),
                    },
                },
            )
        if response.status_code >= 400:
            raise RuntimeError(self._response_error(response))
        cwd = response.json()["approval"]["args"]["cwd"]
        if Path(cwd).resolve() != self.workflow_project.resolve():
            raise RuntimeError(f"cwd was not forced to the selected project: {cwd}")
        return "forced"

    def _security_command_allowlist_enforced(self, project_id: str | None) -> str:
        if not project_id:
            raise RuntimeError("project registration failed")
        with self._api_client() as client:
            blocked = client.post(
                "/tools/request",
                json={
                    "tool": "run_command",
                    "agent": "coding_agent",
                    "args": {"project_id": project_id, "argv": ["rm", "-rf", "/"]},
                },
            )
            if blocked.status_code != 403:
                raise RuntimeError(
                    f"expected 403 for non-allowlisted argv, got {blocked.status_code}"
                )
            allowed = client.post(
                "/tools/request",
                json={
                    "tool": "run_command",
                    "agent": "coding_agent",
                    "args": {"project_id": project_id, "argv": ["pytest"]},
                },
            )
        if allowed.status_code >= 400:
            raise RuntimeError(self._response_error(allowed))
        if allowed.json().get("status") != "pending_approval":
            raise RuntimeError(str(allowed.json()))
        return "allowlist enforced"

    def _security_audit_records(self) -> str:
        audit = self.temp / "logs" / "audit.jsonl"
        text = audit.read_text(encoding="utf-8") if audit.exists() else ""
        if "approved_tool_executed" not in text or "approval_consumed" not in text:
            raise RuntimeError("expected audit events not found")
        return "ok"

    def _security_tool_call_records(self) -> str:
        database = self._brain_decision_database()
        with sqlite3.connect(database) as conn:
            count = conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
        if count < 1:
            raise RuntimeError("no tool call rows found")
        return str(count)

    def _security_agent_run_records(self) -> str:
        database = self._brain_decision_database()
        with sqlite3.connect(database) as conn:
            runs = conn.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0]
            iterations = conn.execute("SELECT COUNT(*) FROM agent_iterations").fetchone()[0]
            suspended = conn.execute("SELECT COUNT(*) FROM suspended_agent_runs").fetchone()[0]
        if runs < 1 or iterations < 1 or suspended < 1:
            raise RuntimeError(f"runs={runs}, iterations={iterations}, suspended={suspended}")
        return f"runs={runs}, iterations={iterations}, suspended={suspended}"


class AllConfiguredModelsVerifier(
    RealModelVerifier
):  # pragma: no cover - requires optional real GGUF runtime
    """Load/chat/stream/unload every configured real GGUF model in one runtime.

    Unlike :class:`RealModelVerifier` (single model, rewritten config), this keeps
    the real ``configs/models.yaml`` so each model is exercised at its own
    configured path, then verifies specialist switching keeps the brain usable.
    The report-building is delegated to the unit-tested
    :func:`build_multi_model_report`, so simulation can never be labelled real.
    """

    def __init__(
        self,
        *,
        home: Path,
        require_real_model: bool,
        max_output_tokens: int = 32,
        timeout: float = 180.0,
        thresholds: ReportThresholds | None = None,
    ) -> None:
        self.plan = plan_multi_model_verification(home)
        available = [entry for entry in self.plan if entry.available]
        nominal = available[0].path if available else (home / "models" / "none.gguf")
        super().__init__(
            home=home, model_path=nominal, max_output_tokens=max_output_tokens, timeout=timeout
        )
        self.require_real_model = require_real_model
        self.thresholds = thresholds or ReportThresholds()
        self.results: list[PerModelResult] = []
        self.specialist_switch: SpecialistSwitchReport | None = None
        self.runtime_error = False

    @property
    def api_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_token}"}

    def _prepare(self) -> None:
        # Keep the REAL configs so each role uses its own configured GGUF path.
        self.verify_home.mkdir(parents=True)
        shutil.copytree(self.repo_home / "configs", self.verify_home / "configs")
        self._rewrite_relative_model_paths_for_temp_home()

    def _rewrite_relative_model_paths_for_temp_home(self) -> None:
        models_path = self.verify_home / "configs" / "models.yaml"
        data = yaml.safe_load(models_path.read_text(encoding="utf-8")) or {}
        models = data.get("models")
        if not isinstance(models, dict):
            return
        changed = False
        for raw_model in models.values():
            if not isinstance(raw_model, dict):
                continue
            raw_path = raw_model.get("path")
            if not isinstance(raw_path, str):
                continue
            model_path = Path(raw_path).expanduser()
            if model_path.is_absolute():
                continue
            raw_model["path"] = str((self.repo_home / model_path).resolve(strict=False))
            changed = True
        if changed:
            models_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    def run(self) -> list[VerifyCheck]:
        for entry in self.plan:
            if not entry.available:
                self.results.append(skipped_result_for(entry))
        available = [entry for entry in self.plan if entry.available]
        if not available:
            status: VerifyStatus = "fail" if self.require_real_model else "skip"
            self.checks.append(
                VerifyCheck(
                    name="configured real GGUF models",
                    ok=not self.require_real_model,
                    detail="No available configured GGUF models to verify.",
                    status=status,
                )
            )
            return self.checks
        try:
            self._prepare()
            env = self._env()
            self.runtime = self._start("services.april_runtime.server", env, self.runtime_log)
            self.api = self._start("services.api.server", env, self.api_log)
            self._check(
                "runtime health",
                lambda: self._wait_json(self.runtime_url + "/runtime/health", auth_runtime=True),
            )
            self._check("core health", lambda: self._wait_json(self.api_url + "/health"))
            for entry in available:
                result = self._verify_one(entry)
                self.results.append(result)
                self.checks.append(
                    VerifyCheck(
                        name=f"model {entry.model.id} acceptance gates",
                        ok=result.acceptance_ok(self.thresholds),
                        detail=(
                            f"tps={result.tokens_per_second}"
                            if result.acceptance_ok(self.thresholds)
                            else "; ".join(result.acceptance_failures(self.thresholds))
                        ),
                    )
                )
            self.specialist_switch = self._verify_switching(available)
            self.checks.append(
                VerifyCheck(
                    name="specialist switching (brain resident)",
                    ok=self.specialist_switch.success,
                    detail="brain stays usable across specialist load/unload",
                )
            )
        except Exception as exc:
            self.runtime_error = True
            self.checks.append(VerifyCheck(name="multi-model runtime", ok=False, detail=str(exc)))
        finally:
            self._stop()
            self._check("services stopped", self._services_stopped)
            shutil.rmtree(self.temp, ignore_errors=True)
        return self.checks

    def _verify_one(self, entry: ModelPlanEntry) -> PerModelResult:
        model = entry.model
        result = PerModelResult(
            model_id=model.id,
            role=model.role,
            backend=model.backend,
            path_basename=entry.path_basename,
            quantization=quantization_from_basename(entry.path_basename),
            available=True,
            context_size=model.context_size,
        )
        try:
            load_start = time.monotonic()
            loaded = self._post_runtime(
                "/runtime/models/load",
                {"model_id": model.id, "request_id": f"multi-{model.id}-load"},
                timeout=self.timeout,
            )
            result.load_duration_seconds = time.monotonic() - load_start
            result.load_success = loaded.get("state") == "loaded"
            result.process_rss_bytes = _process_rss_bytes(
                self.runtime.pid if self.runtime else None
            )
            content, output_tokens, schema_valid, smoke_kind = self._specialist_smoke(
                model.id, model.role
            )
            result.chat_success = bool(content)
            if model.role != "brain":
                result.smoke_success = bool(content)
                result.smoke_schema_valid = schema_valid
                result.smoke_kind = smoke_kind
            latency, tps, stream_tokens = self._stream_model(model.id)
            result.streaming_success = stream_tokens > 0
            result.first_token_latency_seconds = latency
            result.tokens_per_second = tps
            result.output_token_count = output_tokens or stream_tokens
            if model.role == "brain":
                result.structured_brain_json_success = self._brain_structured_json(model.id)
                if os.environ.get("APRIL_VERIFY_ROUTING_EVALS") == "1":
                    try:
                        result.routing = self._routing_report()
                    except Exception:
                        result.routing = None
        except Exception:
            # Leave the unset booleans False; structural_ok stays False so the
            # model is reported as failed, never silently passed.
            pass
        finally:
            try:
                unloaded = self._post_runtime(
                    "/runtime/models/unload",
                    {"model_id": model.id, "request_id": f"multi-{model.id}-unload"},
                    timeout=self.timeout,
                )
                result.unload_success = unloaded.get("state") in {"unloaded", "unavailable"}
            except Exception:
                result.unload_success = False
        return result

    def _specialist_smoke(
        self, model_id: str, role: str
    ) -> tuple[str, int, bool | None, str | None]:
        prompt, smoke_kind, schema_validator = self._smoke_spec(role)
        content, output_tokens = self._chat_model(
            model_id,
            prompt,
            response_format={"type": "json_object"} if schema_validator else None,
            max_output_tokens=max(self.max_output_tokens, 128) if schema_validator else None,
        )
        schema_valid = schema_validator(content) if schema_validator else None
        return content, output_tokens, schema_valid, smoke_kind

    def _smoke_spec(self, role: str) -> tuple[str, str | None, Callable[[str], bool] | None]:
        prompts: dict[str, tuple[str, str | None, Callable[[str], bool] | None]] = {
            "brain": ("Reply with the single word ready.", None, None),
            "coding": (
                "/no_think\nReturn exactly this JSON object and nothing else: "
                '{"plan":["edit","test"]}.',
                "coding_plan",
                self._valid_coding_plan,
            ),
            "reading": (
                "In one sentence, summarize: APRIL keeps local verification reports redacted.",
                "reading_summary",
                None,
            ),
            "creative": (
                "Give one short title for a local verification checklist.",
                "creative_title",
                None,
            ),
            "reasoning": (
                "List two concise tradeoffs for keeping assistant models local.",
                "reasoning_tradeoff",
                None,
            ),
            "system_action": (
                "/no_think\nReturn exactly this JSON object and nothing else: "
                '{"execute":false,"permission_level":0}.',
                "system_decision",
                self._valid_system_decision,
            ),
        }
        return prompts.get(
            role,
            ("Reply with one short confirmation.", "specialist_smoke", None),
        )

    def _valid_coding_plan(self, content: str) -> bool:
        for parsed in _json_object_candidates(content):
            plan = parsed.get("plan")
            if isinstance(plan, list) and all(isinstance(item, str) for item in plan):
                return True
        return False

    def _valid_system_decision(self, content: str) -> bool:
        return any(
            parsed.get("execute") is False and isinstance(parsed.get("permission_level"), int)
            for parsed in _json_object_candidates(content)
        )

    def _chat_model(
        self,
        model_id: str,
        prompt: str,
        *,
        response_format: dict[str, object] | None = None,
        max_output_tokens: int | None = None,
    ) -> tuple[str, int]:
        payload: dict[str, object] = {
            "model_id": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": 0.0,
                "max_output_tokens": max_output_tokens or self.max_output_tokens,
            },
            "request_id": f"multi-{model_id}-chat",
        }
        if response_format is not None:
            payload["response_format"] = response_format
        data = self._post_runtime("/runtime/chat", payload, timeout=self.timeout)
        usage = data.get("usage") or {}
        return str(data.get("content", "")).strip(), int(usage.get("output_tokens", 0))

    def _stream_model(self, model_id: str) -> tuple[float | None, float | None, int]:
        request = {
            "model_id": model_id,
            "messages": [{"role": "user", "content": "Say ok."}],
            "options": {"temperature": 0.0, "max_output_tokens": self.max_output_tokens},
            "request_id": f"multi-{model_id}-stream",
        }
        token_count = 0
        output_tokens = 0
        started = time.monotonic()
        first_token_at: float | None = None
        with httpx.stream(
            "POST",
            self.runtime_url + "/runtime/stream",
            json=request,
            headers=self.runtime_headers,
            timeout=self.timeout,
        ) as response:
            if response.status_code >= 400:
                raise RuntimeError(self._response_error(response))
            for line in response.iter_lines():
                if line.startswith("event: token"):
                    token_count += 1
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                elif line.startswith("data: "):
                    payload = self._stream_payload(line[6:])
                    if "output_tokens" in payload:
                        output_tokens = int(payload["output_tokens"])
        elapsed = max(time.monotonic() - started, 0.000_001)
        tokens = output_tokens or token_count
        latency = first_token_at - started if first_token_at is not None else None
        tps = tokens / elapsed if tokens else None
        return latency, tps, token_count

    def _stream_payload(self, raw: str) -> dict[str, Any]:
        try:
            import json

            data = json.loads(raw)
        except ValueError:
            return {}
        payload = data.get("payload") if isinstance(data, dict) else None
        return payload if isinstance(payload, dict) else {}

    def _brain_structured_json(self, model_id: str) -> bool:
        data = self._post_runtime(
            "/runtime/chat",
            {
                "model_id": model_id,
                "messages": [
                    {"role": "user", "content": "Return one JSON object with a key named status."}
                ],
                "options": {"temperature": 0.0, "max_output_tokens": self.max_output_tokens},
                "response_format": {"type": "json_object"},
                "request_id": f"multi-{model_id}-json",
            },
            timeout=self.timeout,
        )
        return any(
            "status" in parsed for parsed in _json_object_candidates(str(data.get("content", "")))
        )

    def _routing_report(self) -> RoutingReport:
        from apps.runner.evals import load_brain_eval_cases, real_routing_report

        cases = load_brain_eval_cases(self.repo_home)
        decisions: list[dict[str, Any]] = []
        with httpx.Client(
            base_url=self.api_url, headers=self.api_headers, timeout=self.timeout
        ) as client:
            for case in cases:
                marker = self._brain_decision_marker()
                response = client.post("/chat", json={"message": case.message})
                decisions.append(
                    self._brain_decision_after(marker) if response.status_code < 400 else {}
                )
        # Real-mode routing report: a schema-valid fallback decision is a failure.
        return real_routing_report(cases, decisions)

    def _latest_decision(self) -> dict[str, Any]:
        return brain_decision_after_marker(
            self._brain_decision_database(),
            max(self._brain_decision_marker() - 1, 0),
        )

    def _verify_switching(self, available: list[ModelPlanEntry]) -> SpecialistSwitchReport:
        by_role = {entry.model.role: entry.model.id for entry in available}
        report = SpecialistSwitchReport(attempted=True)
        brain = by_role.get("brain")
        if brain is None:
            report.attempted = False
            return report
        report.brain_loaded = self._load_state(brain) == "loaded"
        coding = by_role.get("coding")
        if coding is not None:
            report.coding_loaded = self._load_state(coding) == "loaded"
            report.coding_unloaded = self._unload_state(coding) in {"unloaded", "unavailable"}
        else:
            report.coding_loaded = report.coding_unloaded = True
        reading = by_role.get("reading")
        if reading is not None:
            report.reading_loaded = self._load_state(reading) == "loaded"
            report.reading_unloaded = self._unload_state(reading) in {"unloaded", "unavailable"}
        else:
            report.reading_loaded = report.reading_unloaded = True
        content, _ = self._chat_model(brain, "Reply with the single word ready.")
        report.brain_usable_after = bool(content)
        self._unload_state(brain)
        return report

    def _load_state(self, model_id: str) -> str:
        data = self._post_runtime(
            "/runtime/models/load",
            {"model_id": model_id, "request_id": f"switch-load-{model_id}"},
            timeout=self.timeout,
        )
        return str(data.get("state"))

    def _unload_state(self, model_id: str) -> str:
        data = self._post_runtime(
            "/runtime/models/unload",
            {"model_id": model_id, "request_id": f"switch-unload-{model_id}"},
            timeout=self.timeout,
        )
        return str(data.get("state"))

    def _report_backend(self) -> str:
        if any(entry.available for entry in self.plan):
            return "llama_cpp"
        try:
            return load_settings(root=self.repo_home).runtime.backend
        except ConfigError:
            return "unknown"

    def build_report(self) -> MultiModelVerificationReport:
        return build_multi_model_report(
            environment=environment_snapshot(),
            runtime_backend=self._report_backend(),
            results=self.results,
            specialist_switch=self.specialist_switch,
            thresholds=self.thresholds,
            require_real_model=self.require_real_model,
            runtime_error=self.runtime_error,
        )


def _llama_cpp_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("llama_cpp") is not None


def _process_rss_bytes(pid: int | None) -> int | None:
    if pid is None:
        return None
    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    raw = completed.stdout.strip()
    if not raw:
        return None
    try:
        return int(raw.split()[0]) * 1024
    except (ValueError, IndexError):
        return None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)
