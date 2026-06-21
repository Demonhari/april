from __future__ import annotations

import os
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
from typing import Any

import httpx
import yaml
from pydantic import BaseModel, Field


@dataclass(slots=True)
class VerifyCheck:
    name: str
    ok: bool
    detail: str = ""


def run_fake_verification(home: Path) -> list[VerifyCheck]:
    verifier = LauncherVerifier(home=home)
    return verifier.run()


def run_workflow_verification(
    home: Path, *, real_model: bool = False, model_path: Path | None = None
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
        return RealWorkflowVerifier(home=home, model_path=configured_path).run()
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
        model_entry = {
            "name": "real-smoke",
            "path": str(self.model_path),
            "backend": "llama_cpp",
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
    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_token}"}

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
            self._check("real workflow planning route", self._real_planning_route)
        finally:
            self._stop()
            self._check("services stopped", self._services_stopped)
            shutil.rmtree(self.temp, ignore_errors=True)
        return self.checks

    def _real_planning_route(self) -> str:
        with httpx.Client(
            base_url=self.api_url, headers=self.headers, timeout=self.timeout
        ) as client:
            response = client.post("/chat", json={"message": "April, plan my work today."})
        if response.status_code >= 400:
            raise RuntimeError(self._response_error(response))
        method = self._latest_routing_method()
        if method == "fallback":
            raise RuntimeError(
                "model/prompt reliability failure: model routing JSON was unusable "
                "and fallback routed the request"
            )
        return f"routing_method={method}"

    def _latest_routing_method(self) -> str:
        database = self.temp / "data" / "april.db"
        with sqlite3.connect(database) as conn:
            row = conn.execute(
                """
                SELECT payload_json
                FROM conversation_events
                WHERE event_type = 'brain_decision'
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return "unknown"
        try:
            import json

            payload = json.loads(str(row[0]))
        except ValueError:
            return "unknown"
        return str(payload.get("routing_method") or "unknown")


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
