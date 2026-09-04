from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import yaml

from apps.runner import verify as verify_coordinator
from apps.runner.mac_report import (
    ReportThresholds,
    RoutingReport,
    environment_snapshot,
    quantization_from_basename,
    redact_reason,
)
from apps.runner.multi_model_report import (
    MultiModelVerificationReport,
    PerModelResult,
    SpecialistSwitchReport,
    build_multi_model_report,
)
from apps.runner.verification.models import RealModelVerifier
from apps.runner.verification.types import (
    ModelPlanEntry,
    VerifyCheck,
    VerifyStatus,
)
from april_common.errors import ConfigError
from april_common.settings import load_settings
from services.april_runtime.model_registry import ModelDefinition
from services.brain.schemas import BrainDecision
from services.evolution.adapters import sha256_file


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
        candidate_adapter_model_id: str | None = None,
        candidate_adapter_path: Path | None = None,
        routing_evaluation: bool | None = None,
    ) -> None:
        self.plan = verify_coordinator.plan_multi_model_verification(home)
        available = [entry for entry in self.plan if entry.available]
        nominal = available[0].path if available else (home / "models" / "none.gguf")
        super().__init__(
            home=home, model_path=nominal, max_output_tokens=max_output_tokens, timeout=timeout
        )
        self.require_real_model = require_real_model
        self.routing_evaluation = (
            require_real_model if routing_evaluation is None else routing_evaluation
        )
        self.thresholds = thresholds or ReportThresholds()
        self.results: list[PerModelResult] = []
        self.specialist_switch: SpecialistSwitchReport | None = None
        self.runtime_error = False
        self.candidate_adapter_model_id = candidate_adapter_model_id
        self.candidate_adapter_path = (
            candidate_adapter_path.expanduser().resolve(strict=False)
            if candidate_adapter_path is not None
            else None
        )

    @property
    def api_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_token}"}

    def _prepare(self) -> None:
        # Keep the REAL configs so each role uses its own configured GGUF path.
        self.verify_home.mkdir(parents=True)
        shutil.copytree(self.repo_home / "configs", self.verify_home / "configs")
        self._copy_adapter_pointers_for_temp_home()
        self._rewrite_relative_model_paths_for_temp_home()
        self._inject_candidate_adapter_for_temp_home()

    def _copy_adapter_pointers_for_temp_home(self) -> None:
        source = self.repo_home / "data" / "evolution" / "adapters"
        if not source.is_dir():
            return
        destination = self.verify_home / "data" / "evolution" / "adapters"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)

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
            for key in ("path", "adapter_path"):
                raw_path = raw_model.get(key)
                if not isinstance(raw_path, str):
                    continue
                model_path = Path(raw_path).expanduser()
                if model_path.is_absolute():
                    continue
                raw_model[key] = str((self.repo_home / model_path).resolve(strict=False))
                changed = True
        if changed:
            models_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    def _inject_candidate_adapter_for_temp_home(self) -> None:
        candidate_model_id = getattr(self, "candidate_adapter_model_id", None)
        candidate_path = getattr(self, "candidate_adapter_path", None)
        if candidate_model_id is None or candidate_path is None:
            return
        models_path = self.verify_home / "configs" / "models.yaml"
        data = yaml.safe_load(models_path.read_text(encoding="utf-8")) or {}
        models = data.get("models")
        if not isinstance(models, dict):
            return
        for raw_model in models.values():
            if not isinstance(raw_model, dict):
                continue
            if raw_model.get("id") == candidate_model_id:
                raw_model["adapter_path"] = str(candidate_path)
                models_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
                return

    def run(self) -> list[VerifyCheck]:
        for entry in self.plan:
            if not entry.available:
                self.results.append(verify_coordinator.skipped_result_for(entry))
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
            adapter_path_basename=self._adapter_path_basename(model),
            adapter_sha256=self._adapter_sha256(model),
            available=True,
            context_size=model.context_size,
            routing_evaluation_required=self.routing_evaluation and model.role == "brain",
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
            result.process_rss_bytes = verify_coordinator._process_rss_bytes(
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
                (
                    result.structured_brain_json_success,
                    result.structured_brain_json_fallback,
                    structured_detail,
                ) = self._brain_structured_json_status(model.id)
                if structured_detail:
                    result.failure_detail = structured_detail
                if self.routing_evaluation:
                    try:
                        result.routing = self._routing_report()
                    except Exception as exc:
                        result.routing_error_code = _routing_error_code(exc)
        except Exception as exc:
            # Leave the unset booleans False; structural_ok stays False so the
            # model is reported as failed, never silently passed.
            result.failure_detail = redact_reason(str(exc))[:240]
            if model.role == "brain" and result.structured_brain_json_success is None:
                result.structured_brain_json_success = False
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

    def _effective_adapter_path_for_report(self, model: ModelDefinition) -> Path | None:
        if self.candidate_adapter_model_id == model.id and self.candidate_adapter_path is not None:
            return self.candidate_adapter_path
        try:
            return model.resolved_adapter_path(self.repo_home)
        except (OSError, ValueError):
            return None

    def _adapter_path_basename(self, model: ModelDefinition) -> str | None:
        path = self._effective_adapter_path_for_report(model)
        return path.name if path is not None else None

    def _adapter_sha256(self, model: ModelDefinition) -> str | None:
        path = self._effective_adapter_path_for_report(model)
        if path is None or not path.exists():
            return None
        try:
            return sha256_file(path)
        except OSError:
            return None

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
            "router": (
                "/no_think\nReturn exactly one compact APRIL route JSON object: "
                '{"intent":"planning","agent":"general_agent",'
                '"model_id":"april-brain","confidence":0.7,'
                '"permission_level":0,"risk_level":"none",'
                '"needs_confirmation":false,"decision_summary":"Plan locally."}.',
                "router_decision",
                self._valid_router_decision,
            ),
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
        for parsed in verify_coordinator._json_object_candidates(content):
            plan = parsed.get("plan")
            if isinstance(plan, list) and all(isinstance(item, str) for item in plan):
                return True
        return False

    def _valid_system_decision(self, content: str) -> bool:
        return any(
            parsed.get("execute") is False and isinstance(parsed.get("permission_level"), int)
            for parsed in verify_coordinator._json_object_candidates(content)
        )

    def _valid_router_decision(self, content: str) -> bool:
        return any(
            self._is_valid_brain_decision(candidate)
            for candidate in verify_coordinator._json_object_candidates(content)
        )

    @staticmethod
    def _is_valid_brain_decision(candidate: dict[str, Any]) -> bool:
        try:
            BrainDecision.model_validate(candidate)
        except ValueError:
            return False
        return True

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
        ok, _fallback, _detail = self._brain_structured_json_status(model_id)
        return ok

    def _brain_structured_json_status(self, model_id: str) -> tuple[bool, bool, str | None]:
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
        diagnostics = data.get("diagnostics") if isinstance(data, dict) else {}
        if isinstance(diagnostics, dict) and diagnostics.get("structured_output_fallback") is True:
            reason = diagnostics.get("structured_output_fallback_reason")
            detail = (
                f"structured-output prompt fallback: {reason}"
                if isinstance(reason, str) and reason
                else "structured-output prompt fallback"
            )
            return False, True, detail
        ok = any(
            "status" in parsed
            for parsed in verify_coordinator._json_object_candidates(str(data.get("content", "")))
        )
        return ok, False, None if ok else "response did not contain required status key"

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
        return verify_coordinator.brain_decision_after_marker(
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

    def build_report(
        self, *, config_fingerprint: str | None = None
    ) -> MultiModelVerificationReport:
        return build_multi_model_report(
            environment=environment_snapshot(),
            runtime_backend=self._report_backend(),
            results=self.results,
            specialist_switch=self.specialist_switch,
            thresholds=self.thresholds,
            require_real_model=self.require_real_model,
            runtime_error=self.runtime_error,
            config_fingerprint=config_fingerprint,
        )


def _routing_error_code(exc: BaseException) -> str:
    """Return a bounded, non-secret diagnostic for a failed routing eval."""
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return "routing_timeout"
    if isinstance(exc, httpx.HTTPError):
        return "routing_http_error"
    if isinstance(exc, (OSError, ConnectionError)):
        return "routing_connection_error"
    return "routing_evaluation_failed"
