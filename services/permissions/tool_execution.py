from __future__ import annotations

import base64
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from april_common.errors import PermissionDeniedError
from april_common.project_scope import normalize_project_child, normalize_project_root
from april_common.settings import AprilSettings
from services.memory.schemas import Project
from services.memory.sqlite_memory import SqliteMemory
from services.permissions.approvals import ApprovalStore, canonical_args_hash
from services.permissions.artifacts import (
    build_git_commit_metadata,
    build_patch_approval_metadata,
    load_patch_artifact_bytes,
    verify_approval_artifact,
)
from services.permissions.cleanup import (
    apply_approved_log_cleanup,
    build_log_cleanup_approval_metadata,
)
from services.permissions.engine import PermissionEngine
from services.permissions.schemas import ApprovalRequest, ApprovalResponse, PermissionDecision
from services.permissions.tool_status import ToolCallStatus
from services.tool_worker.client import (
    ToolWorkerClient,
    ToolWorkerProcessManager,
    ToolWorkerUnavailable,
)
from skills.registry import ToolRegistry
from skills.schemas import ToolResult

ExecutionSource = Literal["chat", "orchestrator", "api", "approval", "verify", "cli"]

PROJECT_ROOT_ARGS = {"repo_path", "project_path", "root", "cwd"}
PROJECT_REQUIRED_TOOLS = {
    "git_status",
    "git_diff",
    "git_log",
    "git_branch",
    "git_commit",
    "repo_indexer",
    "test_runner",
    "patch_applier",
    "run_command",
}
PROJECT_OPTIONAL_PATH_TOOLS = {"read_file", "write_file", "list_files", "search_files"}
MAX_STORED_OUTPUT_CHARS = 4000
TOOL_WORKER_REQUIRED_TOOLS = frozenset(
    {"run_command", "test_runner", "patch_applier", "git_commit"}
)


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    request_id: str
    conversation_id: str | None
    actor: str
    agent_id: str
    project_id: str | None
    trusted_project_root: Path | None
    allowed_roots: tuple[Path, ...]
    permission_decision: PermissionDecision | None
    approval_id: str | None
    audit_correlation_id: str
    source: ExecutionSource


@dataclass(frozen=True, slots=True)
class ToolExecutionOutcome:
    status: Literal["executed", "pending_approval", "failed"]
    args: dict[str, Any]
    permission: PermissionDecision
    result: ToolResult | None = None
    approval: ApprovalResponse | None = None


class ToolExecutionService:
    def __init__(
        self,
        *,
        settings: AprilSettings,
        memory: SqliteMemory,
        tool_registry: ToolRegistry,
        permission_engine: PermissionEngine,
        approvals: ApprovalStore,
        tool_worker: ToolWorkerClient | None = None,
    ) -> None:
        self.settings = settings
        self.memory = memory
        self.tool_registry = tool_registry
        self.permission_engine = permission_engine
        self.approvals = approvals
        self.tool_worker = tool_worker
        self._owned_tool_worker_manager: ToolWorkerProcessManager | None = None

    async def aclose(self) -> None:
        if self._owned_tool_worker_manager is not None:
            await self._owned_tool_worker_manager.stop()
            self._owned_tool_worker_manager = None

    async def context(
        self,
        *,
        request_id: str,
        actor: str,
        agent_id: str,
        source: ExecutionSource,
        conversation_id: str | None = None,
        project_id: str | None = None,
        approval_id: str | None = None,
        permission_decision: PermissionDecision | None = None,
    ) -> ToolExecutionContext:
        project: Project | None = None
        if project_id is not None:
            project = await self.memory.get_project(project_id)
            if project is None:
                raise PermissionDeniedError("Project not found.", {"project_id": project_id})
        return ToolExecutionContext(
            request_id=request_id,
            conversation_id=conversation_id,
            actor=actor,
            agent_id=agent_id,
            project_id=project.id if project is not None else None,
            trusted_project_root=normalize_project_root(project.path) if project else None,
            allowed_roots=tuple(self.settings.allowed_roots),
            permission_decision=permission_decision,
            approval_id=approval_id,
            audit_correlation_id=str(uuid.uuid4()),
            source=source,
        )

    async def request_or_execute(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        context: ToolExecutionContext,
        model_permission_level: int = 0,
        model_risk_level: str = "none",
        expected_side_effects: list[str] | None = None,
        approval_metadata: dict[str, Any] | None = None,
    ) -> ToolExecutionOutcome:
        self._refuse_when_disarmed(tool)
        normalized_args = self.normalize_args(tool, args, context)
        permission = self.permission_engine.evaluate(
            tool=tool,
            args=normalized_args,
            agent=context.agent_id,
            model_permission_level=model_permission_level,
            model_risk_level=model_risk_level,
        )
        if (
            permission.risk_level == "external_action"
            and not self.settings.permissions.external_actions_enabled
        ):
            raise PermissionDeniedError("External actions are disabled by configuration.")
        active_context = replace(context, permission_decision=permission)
        if permission.confirmation_required:
            approval = await self.create_approval(
                tool=tool,
                args=normalized_args,
                context=active_context,
                permission=permission,
                expected_side_effects=expected_side_effects,
                metadata_overrides=approval_metadata,
            )
            await self.memory.record_conversation_event(
                conversation_id=context.conversation_id,
                event_type="approval_required",
                payload={
                    "tool": tool,
                    "approval_id": approval.approval_id,
                    "permission_level": permission.permission_level,
                    "risk_level": permission.risk_level,
                },
            )
            return ToolExecutionOutcome(
                status="pending_approval",
                args=normalized_args,
                permission=permission,
                approval=approval,
            )
        result = await self._execute_no_approval(
            tool=tool,
            args=normalized_args,
            context=active_context,
            permission=permission,
        )
        return ToolExecutionOutcome(
            status="executed" if result.ok else "failed",
            args=normalized_args,
            permission=permission,
            result=result,
        )

    async def execute_approved(
        self,
        *,
        approval_id: str,
        actor: str,
        request_id: str,
        tool: str | None = None,
        args: dict[str, Any] | None = None,
    ) -> ToolExecutionOutcome:
        record = await self.approvals.get(approval_id)
        self._refuse_when_disarmed(record.tool)
        if (
            record.risk_level == "external_action"
            and not self.settings.permissions.external_actions_enabled
        ):
            raise PermissionDeniedError("External actions are disabled by configuration.")
        active_tool = tool or record.tool
        active_args = args or record.args
        project_id = str(active_args["project_id"]) if active_args.get("project_id") else None
        context = await self.context(
            request_id=request_id,
            actor=actor,
            agent_id=record.agent,
            source="approval",
            project_id=project_id,
            approval_id=approval_id,
        )
        normalized_args = self.normalize_args(
            active_tool,
            active_args,
            context,
            allow_legacy_approval_repo_path=True,
        )
        approved = await self.approvals.approve_exact(
            approval_id=approval_id,
            tool=active_tool,
            args=normalized_args,
            actor=actor,
            request_id=request_id,
        )
        permission = self.permission_engine.evaluate(
            tool=approved.tool,
            args=approved.args,
            agent=approved.agent,
            model_permission_level=approved.permission_level,
            model_risk_level=approved.risk_level,
        )
        active_context = replace(context, permission_decision=permission)
        self._audit(
            "approved_tool_execution_started",
            active_context,
            approved.tool,
            approved.args,
            "started",
            metadata=approved.metadata,
        )
        precondition_failure = (
            None if approved.tool == "patch_applier" else await verify_approval_artifact(approved)
        )
        if precondition_failure is not None:
            await self._record_tool_call(
                context=active_context,
                tool=approved.tool,
                args=approved.args,
                permission=permission,
                result=precondition_failure,
            )
            await self.approvals.consume(
                approval_id=approval_id,
                result=precondition_failure.model_dump(),
                actor=actor,
                request_id=request_id,
            )
            self._audit(
                "approved_tool_rejected",
                active_context,
                approved.tool,
                approved.args,
                "failed",
                metadata=approved.metadata,
                result=precondition_failure.model_dump(),
            )
            return ToolExecutionOutcome(
                status="failed",
                args=approved.args,
                permission=permission,
                result=precondition_failure,
            )
        try:
            if approved.tool in TOOL_WORKER_REQUIRED_TOOLS:
                result = await self._execute_via_tool_worker(
                    tool=approved.tool,
                    args=approved.args,
                    context=active_context,
                    metadata=approved.metadata,
                )
            elif approved.tool == "apply_log_cleanup":
                result = await apply_approved_log_cleanup(approved)
            else:
                result = await self.tool_registry.execute(approved.tool, approved.args)
        except Exception as exc:
            result = ToolResult(
                ok=False,
                stderr=str(exc),
                risk_level=permission.risk_level,
                permission_level=permission.permission_level,
            )
        result = result.model_copy(
            update={
                "risk_level": permission.risk_level,
                "permission_level": permission.permission_level,
            }
        )
        await self._record_tool_call(
            context=active_context,
            tool=approved.tool,
            args=approved.args,
            permission=permission,
            result=result,
        )
        await self.approvals.consume(
            approval_id=approval_id,
            result=result.model_dump(),
            actor=actor,
            request_id=request_id,
        )
        self._audit(
            "approved_tool_executed",
            active_context,
            approved.tool,
            approved.args,
            "ok" if result.ok else "failed",
            metadata=approved.metadata,
        )
        return ToolExecutionOutcome(
            status="executed" if result.ok else "failed",
            args=approved.args,
            permission=permission,
            result=result,
        )

    def _refuse_when_disarmed(self, tool: str) -> None:
        """During a Dreamer phase, Level >= 1 tools must never execute.

        Every registered tool is at least Level 1, so a disarmed context
        refuses all tool routing outright — analysis phases act on data only.
        """
        from services.evolution.disarm import active_disarmed_phase

        phase = active_disarmed_phase()
        if phase is None:
            return
        definition = self.tool_registry.get(tool)
        level = definition.permission_level if definition is not None else 1
        if level >= 1:
            raise PermissionDeniedError(
                "Tool execution is disarmed during Dreamer phases.",
                {"tool": tool, "phase": phase},
            )

    async def create_approval(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        context: ToolExecutionContext,
        permission: PermissionDecision,
        expected_side_effects: list[str] | None = None,
        metadata_overrides: dict[str, Any] | None = None,
    ) -> ApprovalResponse:
        side_effects = expected_side_effects or self.side_effects(tool)
        metadata = await self.approval_metadata(tool, args, side_effects)
        metadata.update(metadata_overrides or {})
        metadata.setdefault("tool_name", tool)
        metadata.setdefault("canonical_args_hash", canonical_args_hash(args))
        approval = await self.approvals.create(
            ApprovalRequest(
                tool=tool,
                args=args,
                agent=context.agent_id,
                permission_level=permission.permission_level,
                risk_level=permission.risk_level,
                affected_paths=permission.affected_paths,
                expected_side_effects=side_effects,
                metadata=metadata,
            ),
            actor=context.actor,
            request_id=context.request_id,
        )
        return approval

    def normalize_args(
        self,
        tool: str,
        args: dict[str, Any],
        context: ToolExecutionContext,
        *,
        allow_legacy_approval_repo_path: bool = False,
    ) -> dict[str, Any]:
        normalized = dict(args)
        root = context.trusted_project_root
        project_required = tool in PROJECT_REQUIRED_TOOLS
        if project_required and root is None:
            if allow_legacy_approval_repo_path and normalized.get("repo_path"):
                normalized["repo_path"] = str(normalize_project_root(str(normalized["repo_path"])))
                return normalized
            raise PermissionDeniedError(
                "Project-scoped tools require a registered selected project.",
                {"tool": tool},
            )
        if root is None:
            return normalized
        if tool.startswith("git_") or tool in {"repo_indexer", "test_runner", "patch_applier"}:
            normalized["repo_path"] = str(root)
            normalized["project_id"] = context.project_id
        if tool == "run_command":
            normalized["cwd"] = str(root)
        if tool in {"list_files", "search_files"}:
            normalized["path"] = str(
                self._normalize_relative_or_root(
                    normalized.get("path", "."),
                    root,
                    must_exist=True,
                )
            )
        if tool in {"read_file", "write_file"} and "path" in normalized:
            normalized["path"] = str(
                self._normalize_relative_or_root(
                    normalized["path"],
                    root,
                    must_exist=tool == "read_file",
                )
            )
        for key in PROJECT_ROOT_ARGS:
            if (
                key in normalized
                and key not in {"cwd", "repo_path"}
                and tool not in {"run_command"}
            ):
                normalized[key] = str(root)
        return normalized

    async def approval_metadata(
        self, tool: str, args: dict[str, Any], expected_side_effects: list[str]
    ) -> dict[str, Any]:
        if tool == "patch_applier":
            return await build_patch_approval_metadata(
                repo_path=str(args["repo_path"]),
                patch_path=str(args["patch_path"]),
                expected_side_effects=expected_side_effects,
                project_id=str(args["project_id"]) if args.get("project_id") is not None else None,
            )
        if tool == "git_commit":
            return await build_git_commit_metadata(
                repo_path=str(args["repo_path"]),
                message=str(args.get("message")) if args.get("message") is not None else None,
                project_id=str(args["project_id"]) if args.get("project_id") is not None else None,
            )
        if tool == "apply_log_cleanup":
            return await build_log_cleanup_approval_metadata(
                manifest_id=str(args["manifest_id"]),
                expected_side_effects=expected_side_effects,
            )
        return {}

    def side_effects(self, tool: str) -> list[str]:
        if tool == "patch_applier":
            return ["Apply a local patch to repository files."]
        if tool == "run_command":
            return ["Run a configured local developer command."]
        if tool == "git_commit":
            return ["Create a local Git commit."]
        if tool == "repo_indexer":
            return ["Update APRIL's local repository index."]
        if tool == "apply_log_cleanup":
            return ["Delete exactly the files in an approved local cleanup manifest."]
        return ["Perform a restricted local action."]

    async def _execute_no_approval(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        context: ToolExecutionContext,
        permission: PermissionDecision,
    ) -> ToolResult:
        try:
            if tool in TOOL_WORKER_REQUIRED_TOOLS:
                result = await self._execute_via_tool_worker(
                    tool=tool,
                    args=args,
                    context=context,
                    metadata={},
                )
            else:
                result = await self.tool_registry.execute(tool, args)
        except Exception as exc:
            result = ToolResult(
                ok=False,
                stderr=str(exc),
                risk_level=permission.risk_level,
                permission_level=permission.permission_level,
            )
        result = result.model_copy(
            update={
                "risk_level": permission.risk_level,
                "permission_level": permission.permission_level,
            }
        )
        await self._record_tool_call(
            context=context,
            tool=tool,
            args=args,
            permission=permission,
            result=result,
        )
        if permission.permission_level >= 2:
            self._audit(
                "tool_executed",
                context,
                tool,
                args,
                "ok" if result.ok else "failed",
            )
        return result

    async def _execute_via_tool_worker(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        context: ToolExecutionContext,
        metadata: dict[str, Any],
    ) -> ToolResult:
        permission = context.permission_decision
        risk = permission.risk_level if permission is not None else "code_write"
        level = permission.permission_level if permission is not None else 3
        if tool == "patch_applier" and (
            metadata.get("artifact_type") != "patch"
            or not metadata.get("artifact_id")
            or not metadata.get("patch_sha256")
        ):
            return ToolResult(
                ok=False,
                stderr="Patch approval is missing immutable artifact metadata.",
                data={"failure_code": "patch_artifact_metadata_missing"},
                risk_level=risk,
                permission_level=level,
            )
        if self.tool_worker is None:
            manager = ToolWorkerProcessManager(
                april_home=self.settings.home,
                allowed_roots=tuple(self.settings.allowed_roots),
            )
            try:
                self.tool_worker = await manager.start()
                self._owned_tool_worker_manager = manager
            except (OSError, ToolWorkerUnavailable):
                return ToolResult(
                    ok=False,
                    stderr="Tool Worker is unavailable.",
                    data={"failure_code": "tool_worker_unavailable"},
                    risk_level=risk,
                    permission_level=level,
                )
        root_value = args.get("repo_path") or args.get("cwd")
        if root_value is None:
            return ToolResult(
                ok=False,
                stderr="Tool Worker request has no trusted project root.",
                data={"failure_code": "worker_project_root_missing"},
                risk_level=risk,
                permission_level=level,
            )
        worker_args: dict[str, object]
        if tool in {"run_command", "test_runner"}:
            worker_args = {"argv": list(args.get("argv", []))}
        elif tool == "patch_applier":
            artifact_id = str(metadata.get("artifact_id", ""))
            try:
                patch_bytes = load_patch_artifact_bytes(artifact_id)
            except PermissionDeniedError:
                return ToolResult(
                    ok=False,
                    stderr="Approved patch artifact is unavailable.",
                    data={"failure_code": "patch_artifact_unavailable"},
                    risk_level=risk,
                    permission_level=level,
                )
            worker_args = {
                "patch_base64": base64.b64encode(patch_bytes).decode("ascii"),
                "patch_sha256": str(metadata.get("patch_sha256", "")),
                "patch_byte_length": metadata.get("patch_byte_length"),
                "affected_paths": list(metadata.get("affected_paths", [])),
                "repo_root": str(metadata.get("repo_root", "")),
                "repo_state_digest": metadata.get("repo_state_digest"),
            }
        elif tool == "git_commit":
            worker_args = {
                "message": str(args.get("message", "")),
                "staged_diff_sha256": str(metadata.get("staged_diff_sha256", "")),
                "staged_tree_id": str(metadata.get("staged_tree_id", "")),
            }
        else:
            return ToolResult(
                ok=False,
                stderr="Tool Worker operation is unsupported.",
                data={"failure_code": "worker_operation_unsupported"},
                risk_level=risk,
                permission_level=level,
            )
        definition = self.tool_registry.get(tool)
        timeout = (
            definition.timeout_seconds
            if definition is not None
            else self.settings.permissions.tool_timeout_seconds
        )
        try:
            response = await self.tool_worker.execute(
                request_id=f"{context.request_id}:{tool}",
                operation=tool,
                project_root=normalize_project_root(str(root_value)),
                args=worker_args,
                timeout_seconds=timeout,
            )
        except ToolWorkerUnavailable:
            return ToolResult(
                ok=False,
                stderr="Tool Worker is unavailable.",
                data={"failure_code": "tool_worker_unavailable"},
                risk_level=risk,
                permission_level=level,
            )
        data = dict(response.data)
        if response.failure_code:
            data["failure_code"] = response.failure_code
        if response.stdout_truncated:
            data["stdout_truncated"] = True
        if response.stderr_truncated:
            data["stderr_truncated"] = True
        return ToolResult(
            ok=response.ok,
            stdout=response.stdout,
            stderr=response.stderr,
            data=data,
            risk_level=risk,
            permission_level=level,
        )

    async def _record_tool_call(
        self,
        *,
        context: ToolExecutionContext,
        tool: str,
        args: dict[str, Any],
        permission: PermissionDecision,
        result: ToolResult,
    ) -> None:
        await self.memory.record_tool_call(
            tool=tool,
            args=self._sanitize_tool_args(tool, args),
            status=(ToolCallStatus.EXECUTED.value if result.ok else ToolCallStatus.FAILED.value),
            permission_level=permission.permission_level,
            risk_level=permission.risk_level,
            result=self._sanitize_result(result),
            conversation_id=context.conversation_id,
        )

    def _audit(
        self,
        event_type: str,
        context: ToolExecutionContext,
        tool: str,
        args: dict[str, Any],
        outcome: str,
        *,
        metadata: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        permission = context.permission_decision
        self.approvals.audit.write(
            {
                "actor": context.actor,
                "request_id": context.request_id,
                "audit_correlation_id": context.audit_correlation_id,
                "event_type": event_type,
                "tool": tool,
                "arguments": self._sanitize_tool_args(tool, args),
                "agent": context.agent_id,
                "project_id": context.project_id,
                "approval_id": context.approval_id,
                "permission_level": permission.permission_level if permission else None,
                "risk": permission.risk_level if permission else None,
                "metadata": metadata or {},
                "result": self._sanitize_mapping(result or {}),
                "outcome": outcome,
            }
        )

    def _normalize_relative_or_root(self, value: object, root: Path, *, must_exist: bool) -> Path:
        raw = str(value)
        requested = Path(raw).expanduser()
        if requested.is_absolute():
            raise PermissionDeniedError("Project-scoped model paths must be relative.")
        return normalize_project_child(requested, project_root=root, must_exist=must_exist)

    def _sanitize_result(self, result: ToolResult) -> dict[str, Any]:
        data = result.model_dump()
        data["stdout"] = self._truncate_secret_text(str(data.get("stdout", "")))
        data["stderr"] = self._truncate_secret_text(str(data.get("stderr", "")))
        data["data"] = self._sanitize_mapping(data.get("data", {}))
        return data

    def _sanitize_tool_args(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        sanitized = self._sanitize_mapping(args)
        if tool == "remember_memory" and isinstance(sanitized, dict):
            sanitized["content"] = "[REDACTED]"
        return sanitized

    def _sanitize_mapping(self, value: Any) -> Any:
        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            for key, item in value.items():
                lowered = str(key).lower()
                if any(secret in lowered for secret in ("token", "secret", "password", "key")):
                    sanitized[str(key)] = "[REDACTED]"
                else:
                    sanitized[str(key)] = self._sanitize_mapping(item)
            return sanitized
        if isinstance(value, list):
            return [self._sanitize_mapping(item) for item in value]
        if isinstance(value, str):
            return self._truncate_secret_text(value)
        return value

    def _truncate_secret_text(self, value: str) -> str:
        if "-----BEGIN" in value or "authorization:" in value.lower():
            return "[REDACTED]"
        if len(value) > MAX_STORED_OUTPUT_CHARS:
            return value[:MAX_STORED_OUTPUT_CHARS] + "\n[TRUNCATED]"
        return value
