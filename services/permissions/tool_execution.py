from __future__ import annotations

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
    apply_approved_patch,
    build_git_commit_metadata,
    build_patch_approval_metadata,
    verify_approval_artifact,
)
from services.permissions.engine import PermissionEngine
from services.permissions.schemas import ApprovalRequest, ApprovalResponse, PermissionDecision
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
    ) -> None:
        self.settings = settings
        self.memory = memory
        self.tool_registry = tool_registry
        self.permission_engine = permission_engine
        self.approvals = approvals

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
            if approved.tool == "patch_applier":
                result = await apply_approved_patch(approved)
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
            args=self._sanitize_mapping(args),
            status="ok" if result.ok else "failed",
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
                "arguments": self._sanitize_mapping(args),
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
