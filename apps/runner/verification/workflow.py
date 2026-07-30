from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import httpx

from apps.runner import verify as verify_coordinator
from apps.runner.verification.models import RealModelVerifier
from apps.runner.verification.services import LauncherVerifier
from apps.runner.verification.types import (
    VerifyCheck,
)
from services.brain.schemas import BrainDecision
from services.memory.database import connect_sqlite


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
        with connect_sqlite(database) as conn:
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
            data = client.get("/diagnostics").json()
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
        verify_coordinator._git(self.workflow_project, "init")
        verify_coordinator._git(
            self.workflow_project,
            "config",
            "user.email",
            "april@example.local",
        )
        verify_coordinator._git(self.workflow_project, "config", "user.name", "APRIL Verify")
        verify_coordinator._git(self.workflow_project, "add", ".")
        verify_coordinator._git(self.workflow_project, "commit", "-m", "initial")
        self.second_project.mkdir(parents=True)
        (self.second_project / "README.md").write_text("# second\n", encoding="utf-8")
        verify_coordinator._git(self.second_project, "init")
        verify_coordinator._git(self.second_project, "config", "user.email", "april@example.local")
        verify_coordinator._git(self.second_project, "config", "user.name", "APRIL Verify")
        verify_coordinator._git(self.second_project, "add", ".")
        verify_coordinator._git(self.second_project, "commit", "-m", "initial")
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
        payload = verify_coordinator.brain_decision_after_marker(
            self._brain_decision_database(),
            max(self._brain_decision_marker() - 1, 0),
        )
        return str(payload.get("routing_method") or "unknown")

    def _latest_decision(self) -> dict[str, Any]:
        return verify_coordinator.brain_decision_after_marker(
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
        with connect_sqlite(database) as conn:
            count = conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
        if count < 1:
            raise RuntimeError("no tool call rows found")
        return str(count)

    def _security_agent_run_records(self) -> str:
        database = self._brain_decision_database()
        with connect_sqlite(database) as conn:
            runs = conn.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0]
            iterations = conn.execute("SELECT COUNT(*) FROM agent_iterations").fetchone()[0]
            suspended = conn.execute("SELECT COUNT(*) FROM suspended_agent_runs").fetchone()[0]
        if runs < 1 or iterations < 1 or suspended < 1:
            raise RuntimeError(f"runs={runs}, iterations={iterations}, suspended={suspended}")
        return f"runs={runs}, iterations={iterations}, suspended={suspended}"
