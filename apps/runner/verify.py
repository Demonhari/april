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


@dataclass(slots=True)
class VerifyCheck:
    name: str
    ok: bool
    detail: str = ""


def run_fake_verification(home: Path) -> list[VerifyCheck]:
    verifier = LauncherVerifier(home=home)
    return verifier.run()


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
                response = httpx.get(url, timeout=1.0)
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
        patch_dir = self.verify_home / "data" / "patches"
        patch_dir.mkdir(parents=True, exist_ok=True)
        patch_path = patch_dir / "verify.patch"
        patch_path.write_text(
            "diff --git a/app.py b/app.py\n"
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1 +1 @@\n"
            "-value = 'old'\n"
            "+value = 'new'\n",
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
        return str(response["approval"]["approval_id"])

    def _approve(self, approval_id: str) -> str:
        with self._client() as client:
            response = client.post("/tools/approve", json={"approval_id": approval_id}).json()
        if response.get("status") != "executed":
            raise RuntimeError(str(response))
        if "value = 'new'" not in (self.project / "app.py").read_text(encoding="utf-8"):
            raise RuntimeError("patch was not applied")
        return "applied"

    def _approval_replay_rejected(self, approval_id: str) -> str:
        with self._client() as client:
            response = client.post("/tools/approve", json={"approval_id": approval_id})
        if response.status_code != 403:
            raise RuntimeError(f"expected 403, got {response.status_code}")
        return "403"

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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)
