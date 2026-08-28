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

from apps.runner import verify as verify_coordinator
from apps.runner.verification.types import (
    MissingChatResultError,
    VerifyCheck,
)
from april_common.process_environment import ProcessCategory, build_process_environment
from april_common.service_health import ServiceHealthResult, probe_service_health
from services.memory.database import connect_sqlite, sqlite_write_transaction


class LauncherVerifier:
    def __init__(self, *, home: Path, development_unsandboxed_override: bool = False) -> None:
        self.repo_home = home.expanduser().resolve()
        self.temp = Path(tempfile.mkdtemp(prefix="april-verify-"))
        self.verify_home = self.temp / "april_home"
        self.project = self.temp / "external_project"
        self.second_project = self.temp / "second_project"
        self.runtime_port = verify_coordinator._free_port()
        self.api_port = verify_coordinator._free_port()
        self.api_token = "verify-token"
        self.runtime_token = "verify-runtime-token"
        self.development_unsandboxed_override = development_unsandboxed_override
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
            self._check("core readiness", self._core_readiness)
            self._check("durable job self-check", self._job_self_check)
            self._check("model listing", self._check_models)
            project_id = self._check("project registration", self._register_project)
            # Unscored readiness warm-up: import contention can make the first
            # tool-routing chat return before the normal result envelope exists.
            # Its outcome is deliberately ignored and never changes totals.
            self._warm_up_tool_routing(project_id)
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
        verify_coordinator._git(self.project, "init")
        verify_coordinator._git(self.project, "config", "user.email", "april@example.local")
        verify_coordinator._git(self.project, "config", "user.name", "APRIL Verify")
        verify_coordinator._git(self.project, "add", ".")
        verify_coordinator._git(self.project, "commit", "-m", "initial")

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
                "APRIL_RUNTIME_BACKEND": "fake",
                "APRIL_DEVELOPMENT_UNSANDBOXED_OVERRIDE": (
                    "true" if self.development_unsandboxed_override else "false"
                ),
                "APRIL_RUNTIME_PORT": str(self.runtime_port),
                "APRIL_API_PORT": str(self.api_port),
                "APRIL_RUNTIME_URL": self.runtime_url,
                **credential_environment,
                "APRIL_DATABASE_PATH": str(self.temp / "data" / "april.db"),
                "APRIL_VECTOR_INDEX_PATH": str(self.temp / "data" / "vector_index"),
                "APRIL_AUDIT_PATH": str(self.temp / "logs" / "audit.jsonl"),
                "APRIL_LOGS_PATH": str(self.temp / "logs"),
                "APRIL_ALLOWED_FILESYSTEM_ROOTS": f"{self.project},{self.second_project}",
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
                start_new_session=True,
            )

    def _wait_json(self, url: str) -> dict[str, Any]:
        deadline = time.monotonic() + 20.0
        last = ServiceHealthResult(False, None, "connection_failed", "Endpoint is not reachable.")
        while time.monotonic() < deadline:
            token = self.runtime_token if url.startswith(self.runtime_url) else None
            last = probe_service_health(url, bearer_token=token, timeout=1.0)
            if last.ok:
                return {"status": "ok", "http_status": last.status_code}
            time.sleep(0.2)
        raise RuntimeError(verify_coordinator._verification_health_failure(url, self.api_url, last))

    def _core_readiness(self) -> str:
        try:
            response = httpx.get(
                self.api_url + "/readiness",
                headers=self.headers,
                timeout=2.0,
                follow_redirects=False,
            )
        except httpx.HTTPError as exc:
            raise RuntimeError("Core API is alive but readiness could not be read.") from exc
        if response.status_code != 200:
            raise RuntimeError(
                f"Core API is alive but readiness returned HTTP {response.status_code}."
            )
        payload = response.json()
        if payload.get("ready") is not True:
            raw_reasons = payload.get("failure_reasons")
            reasons = raw_reasons if isinstance(raw_reasons, list) else []
            details = "; ".join(
                f"{reason.get('code', 'not_ready')}: {reason.get('message', 'not ready')}"
                for reason in reasons
                if isinstance(reason, dict)
            )
            raise RuntimeError(
                "Core API is alive but not ready" + (f": {details}" if details else ".")
            )
        tool_worker = payload.get("tool_worker", {})
        jobs = payload.get("jobs", {})
        if not isinstance(tool_worker, dict) or tool_worker.get("self_check") is not True:
            raise RuntimeError("Tool Worker readiness self-check failed.")
        if not isinstance(jobs, dict) or jobs.get("worker_readiness") is not True:
            raise RuntimeError("Job Worker is not ready.")
        return "ready with Tool Worker and Job Worker"

    def _job_self_check(self) -> str:
        with self._client(timeout=5.0) as client:
            response = client.post(
                "/jobs",
                json={"job_type": "self_check", "payload": {}},
            )
            if response.status_code != 200:
                raise RuntimeError(f"Job submission returned HTTP {response.status_code}.")
            job_id = str(response.json().get("id", ""))
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                shown = client.get(f"/jobs/{job_id}")
                if shown.status_code != 200:
                    raise RuntimeError(f"Job inspection returned HTTP {shown.status_code}.")
                payload = shown.json()
                status = str(payload.get("status", ""))
                if status == "succeeded":
                    result = payload.get("result")
                    if not isinstance(result, dict) or result.get("self_check") is not True:
                        raise RuntimeError("Job self-check result was malformed.")
                    return "submitted, claimed, and durably completed"
                if status in {"cancelled", "failed", "interrupted"}:
                    raise RuntimeError(f"Job self-check ended as {status}.")
                time.sleep(0.1)
        raise RuntimeError("Job self-check did not complete before timeout.")

    def _client(self, *, timeout: float = 10.0) -> httpx.Client:
        return httpx.Client(base_url=self.api_url, headers=self.headers, timeout=timeout)

    def _warm_up_tool_routing(self, project_id: str) -> None:
        try:
            with self._client(timeout=30.0) as client:
                client.post(
                    "/chat",
                    json={
                        "message": "April, inspect this repository for a routing warm-up.",
                        "project_id": project_id,
                    },
                )
        except Exception:
            return

    def _post_chat_result(
        self,
        client: httpx.Client,
        payload: dict[str, Any],
        *,
        context: str,
        retry_missing_result: bool = False,
    ) -> dict[str, Any]:
        attempts = 2 if retry_missing_result else 1
        for attempt in range(attempts):
            response = client.post("/chat", json=payload)
            try:
                return verify_coordinator.chat_result_from_response(response, context=context)
            except MissingChatResultError:
                if attempt + 1 >= attempts:
                    raise
        raise AssertionError("unreachable")

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
            first = self._post_chat_result(
                client,
                {"message": "April, plan my work today.", "project_id": project_id},
                context="first scored tool-routing chat",
                retry_missing_result=True,
            )
            conversation_id = first["conversation_id"]
            second = self._post_chat_result(
                client,
                {
                    "message": "Use that same plan.",
                    "project_id": project_id,
                    "conversation_id": conversation_id,
                },
                context="second conversation turn",
            )
        if second["status"] != "ok":
            raise RuntimeError("second turn failed")
        return str(conversation_id)

    def _isolated_conversation(self, project_id: str, existing_id: str) -> str:
        with self._client() as client:
            other = self._post_chat_result(
                client,
                {"message": "Start a separate plan.", "project_id": project_id},
                context="conversation isolation chat",
            )
        other_id = other["conversation_id"]
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
            result = self._post_chat_result(
                client,
                {
                    "message": "April, check why the animation in this repository is broken.",
                    "project_id": project_id,
                },
                context="read-only repo analysis",
            )
        if result["status"] != "ok":
            raise RuntimeError("repo analysis failed")
        return "ok"

    def _patch_approval(self, project_id: str) -> str:
        with self._client() as client:
            result = self._post_chat_result(
                client,
                {
                    "message": "Apply the fix.",
                    "project_id": project_id,
                },
                context="patch approval chat",
            )
        if result["status"] != "pending_approval":
            raise RuntimeError(str(result))
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
            )
        result = verify_coordinator.chat_result_from_response(
            response, context="direct agent structured execution"
        )
        if result["status"] != "ok":
            raise RuntimeError(str(result))
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
            with sqlite_write_transaction(database) as conn:
                conn.execute(
                    "UPDATE approvals SET expires_at = ? WHERE id = ?",
                    ("1970-01-01T00:00:00Z", approval_id),
                )
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
        deadline = time.monotonic() + 5.0
        last_count = 0
        while True:
            with connect_sqlite(database) as conn:
                last_count = conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
            if last_count >= 1:
                return str(last_count)
            if time.monotonic() >= deadline:
                raise RuntimeError("no tool call rows found")
            time.sleep(0.1)

    def _agent_run_records(self) -> str:
        database = self.temp / "data" / "april.db"
        deadline = time.monotonic() + 5.0
        runs = iterations = suspended = 0
        while True:
            with connect_sqlite(database) as conn:
                runs = conn.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0]
                iterations = conn.execute("SELECT COUNT(*) FROM agent_iterations").fetchone()[0]
                suspended = conn.execute("SELECT COUNT(*) FROM suspended_agent_runs").fetchone()[0]
                route_sources = [
                    str(row[0])
                    for row in conn.execute(
                        """
                        SELECT DISTINCT json_extract(metadata_json, '$.route_source')
                        FROM agent_runs
                        WHERE json_extract(metadata_json, '$.route_source') IS NOT NULL
                        ORDER BY 1
                        """
                    ).fetchall()
                ]
            if runs >= 1 and iterations >= 1 and suspended >= 1:
                sources = ",".join(route_sources) if route_sources else "none"
                return (
                    f"runs={runs}, iterations={iterations}, suspended={suspended}, "
                    f"route_sources={sources}"
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(f"runs={runs}, iterations={iterations}, suspended={suspended}")
            time.sleep(0.1)

    def _suspended_status(self, approval_id: str) -> str | None:
        database = self.temp / "data" / "april.db"
        if not database.exists():
            return None
        with connect_sqlite(database) as conn:
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
