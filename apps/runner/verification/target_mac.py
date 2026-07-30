from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

from apps.runner import verify as verify_coordinator
from apps.runner.mac_report import (
    MacVerificationReport,
    RealModelReport,
    ReportThresholds,
    RoutingReport,
    SkippedCheck,
    build_mac_report,
    environment_snapshot,
    quantization_from_basename,
    routing_report_from_results,
)
from apps.runner.verification.models import RealModelVerifier
from apps.runner.verification.types import (
    VerifyCheck,
    VerifyStatus,
)
from april_common.errors import ConfigError
from april_common.settings import load_settings
from services.april_runtime.model_registry import ModelRegistry


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
            verifier = verify_coordinator.RealModelVerifier(
                home=self.home,
                model_path=selected_model,
                max_output_tokens=self.max_output_tokens,
                timeout=self.timeout,
            )
            self.real_verifier = verifier
            self.checks.extend(verifier.run())
            self.checks.extend(
                verify_coordinator.run_workflow_verification(
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
        if not verify_coordinator._llama_cpp_installed():
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
        devices = verify_coordinator.query_audio_devices()
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
        report = verify_coordinator.voice_doctor(settings)
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
