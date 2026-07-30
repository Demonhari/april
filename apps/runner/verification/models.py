from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx
import yaml

from apps.runner import verify as verify_coordinator
from apps.runner.verification.types import (
    BenchmarkResult,
    VerifyCheck,
)
from april_common.process_environment import ProcessCategory, build_process_environment
from april_common.service_health import ServiceHealthResult, probe_service_health
from april_common.thermal_state import ThermalStateResult, collect_thermal_state
from april_common.time import utc_now_iso


class RealModelVerifier:  # pragma: no cover - requires optional real GGUF runtime
    def __init__(
        self,
        *,
        home: Path,
        model_path: Path,
        max_output_tokens: int = 32,
        timeout: float = 180.0,
        inherit_process_group: bool = False,
    ) -> None:
        self.repo_home = home.expanduser().resolve()
        self.model_path = model_path.expanduser().resolve()
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.inherit_process_group = inherit_process_group
        self.temp = Path(tempfile.mkdtemp(prefix="april-real-verify-"))
        self.verify_home = self.temp / "april_home"
        self.runtime_port = verify_coordinator._free_port()
        self.api_port = verify_coordinator._free_port()
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
        self.prompt_token_count: int | None = None
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
        chat_format = verify_coordinator._infer_chat_format_from_basename(self.model_path.name)
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
        credential_environment = verify_coordinator._verification_credential_environment(
            verify_home=self.verify_home,
            temporary_root=self.temp,
            api_token=self.api_token,
            runtime_token=self.runtime_token,
        )
        return build_process_environment(
            ProcessCategory.VERIFICATION_SUBPROCESS,
            april_home=self.verify_home,
            overrides={
                "APRIL_HOME": str(self.verify_home),
                "PYTHONPATH": str(self.repo_home),
                "APRIL_RUNTIME_BACKEND": "llama_cpp",
                "APRIL_RUNTIME_PRELOAD_KEEP_LOADED": "false",
                "APRIL_RUNTIME_PORT": str(self.runtime_port),
                "APRIL_API_PORT": str(self.api_port),
                "APRIL_RUNTIME_URL": self.runtime_url,
                **credential_environment,
                "APRIL_DATABASE_PATH": str(self.temp / "data" / "april.db"),
                "APRIL_VECTOR_INDEX_PATH": str(self.temp / "data" / "vector_index"),
                "APRIL_AUDIT_PATH": str(self.temp / "logs" / "audit.jsonl"),
                "APRIL_LOGS_PATH": str(self.temp / "logs"),
                "APRIL_ALLOWED_FILESYSTEM_ROOTS": str(self.temp),
            },
        )

    def _start(self, module: str, env: dict[str, str], log_path: Path) -> subprocess.Popen[bytes]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        category = (
            ProcessCategory.RUNTIME
            if module == "services.april_runtime.server"
            else ProcessCategory.CORE_API
        )
        child_env = build_process_environment(
            category,
            source=env,
            april_home=Path(env["APRIL_HOME"]),
        )
        with log_path.open("ab") as log_file:
            return subprocess.Popen(
                [sys.executable, "-m", module],
                cwd=str(self.repo_home),
                env=child_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=not self.inherit_process_group,
            )

    def _wait_json(self, url: str, *, auth_runtime: bool = False) -> dict[str, Any]:
        deadline = time.monotonic() + 30.0
        last = ServiceHealthResult(False, None, "connection_failed", "Endpoint is not reachable.")
        while time.monotonic() < deadline:
            token = self.runtime_token if auth_runtime else None
            last = probe_service_health(url, bearer_token=token, timeout=1.0)
            if last.ok:
                return {"status": "ok", "http_status": last.status_code}
            time.sleep(0.2)
        raise RuntimeError(verify_coordinator._verification_health_failure(url, self.api_url, last))

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
        self.runtime_rss_bytes = verify_coordinator._process_rss_bytes(
            self.runtime.pid if self.runtime else None
        )
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
        if "input_tokens" in payload:
            self.prompt_token_count = int(payload["input_tokens"])
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
        return verify_coordinator.latest_brain_decision_marker(self._brain_decision_database())

    def _brain_decision_after(self, marker: int) -> dict[str, Any]:
        return verify_coordinator.brain_decision_after_marker(
            self._brain_decision_database(), marker
        )

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
                with suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGTERM)
        for proc in (self.api, self.runtime):
            if proc is None:
                continue
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
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
        inherit_process_group: bool = False,
        thermal_collector: Callable[[], ThermalStateResult] = collect_thermal_state,
    ) -> None:
        super().__init__(
            home=home,
            model_path=model_path,
            max_output_tokens=max_output_tokens,
            timeout=180.0,
            inherit_process_group=inherit_process_group,
        )
        self.prompt = prompt
        self.runs = runs
        self.keep_loaded = keep_loaded
        self.thermal_collector = thermal_collector
        self.thermal_samples: list[ThermalStateResult] = []

    def run(self) -> list[BenchmarkResult]:  # type: ignore[override]
        return self.run_with_evaluation()[0]

    def run_with_evaluation(
        self,
        evaluator: Callable[[ModelBenchmark], dict[str, Any]] | None = None,
    ) -> tuple[list[BenchmarkResult], dict[str, Any] | None]:
        results: list[BenchmarkResult] = []
        evaluation: dict[str, Any] | None = None
        self._sample_thermal()
        try:
            self._prepare()
            env = self._env()
            self.runtime = self._start("services.april_runtime.server", env, self.runtime_log)
            self.api = self._start("services.api.server", env, self.api_log)
            self._wait_json(self.runtime_url + "/runtime/health", auth_runtime=True)
            self._wait_json(self.api_url + "/health")
            self._sample_thermal()
            for index in range(1, self.runs + 1):
                results.append(self._run_one(index))
                if index < self.runs:
                    self._sample_thermal()
            if evaluator is not None:
                evaluation = evaluator(self)
        finally:
            self._sample_thermal()
            self._stop()
            shutil.rmtree(self.temp, ignore_errors=True)
        return results, evaluation

    def _sample_thermal(self) -> None:
        try:
            self.thermal_samples.append(self.thermal_collector())
        except Exception:
            # A diagnostic probe must never abort or relabel the benchmark.
            self.thermal_samples.append(
                ThermalStateResult(
                    available=False,
                    sampled_at=utc_now_iso(),
                    failure_reason="collector_failed",
                )
            )

    def _run_one(self, index: int) -> BenchmarkResult:
        self.load_time_seconds = None
        self.first_token_latency_seconds = None
        self.generation_time_seconds = None
        self.output_tokens = 0
        self.tokens_per_second = None
        self.prompt_token_count = None
        try:
            self._load_model()
            cold_load_time = self.load_time_seconds or 0.0
            self._load_model()
            warm_load_time = self.load_time_seconds
            self._benchmark_stream()
            process_rss = verify_coordinator._process_rss_bytes(
                self.runtime.pid if self.runtime else None
            )
            peak_process_rss: int | None = None
            try:
                health = self._wait_json(
                    self.runtime_url + "/runtime/health",
                    auth_runtime=True,
                )
                peak_value = health.get("process_peak_rss_bytes")
                if isinstance(peak_value, int):
                    peak_process_rss = peak_value
            except Exception:
                peak_process_rss = None
            unload_success = False
            unload_time: float | None = None
            if not self.keep_loaded:
                unload_started = time.monotonic()
                self._unload_model()
                unload_time = time.monotonic() - unload_started
                unload_success = True
            return BenchmarkResult(
                run_index=index,
                ok=True,
                load_time_seconds=cold_load_time,
                warm_load_time_seconds=warm_load_time,
                first_token_latency_seconds=self.first_token_latency_seconds,
                generation_time_seconds=self.generation_time_seconds or 0.0,
                output_tokens=self.output_tokens,
                tokens_per_second=self.tokens_per_second or 0.0,
                unload_success=unload_success,
                unload_time_seconds=unload_time,
                process_rss_bytes=process_rss,
                peak_process_rss_bytes=peak_process_rss,
                prompt_token_count=self.prompt_token_count,
                prompt_eval_duration_seconds=self.first_token_latency_seconds,
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
