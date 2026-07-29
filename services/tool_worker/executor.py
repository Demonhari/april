from __future__ import annotations

import asyncio
import base64
import hmac
from pathlib import Path
from typing import Any

from april_common.errors import PermissionDeniedError
from april_common.process_environment import ProcessCategory
from april_common.process_runner import (
    ProcessStatus,
    ResourceLimitProfile,
    run_restricted_process,
)
from april_common.process_sandbox import operation_policy
from april_common.project_scope import (
    git_apply_bytes,
    git_apply_check_bytes,
    git_staged_digest,
    git_staged_tree_id,
    git_worktree_digest,
    inspect_patch_bytes,
    normalize_project_root,
)
from services.tool_worker.schemas import ToolWorkerRequest, ToolWorkerResponse
from skills.git.common import run_git
from skills.terminal.command_policy import validate_command


class ToolWorkerExecutor:
    def __init__(
        self,
        *,
        allowed_roots: tuple[Path, ...],
        capability: str,
        environment: str | None = None,
        development_unsandboxed_override: bool = False,
    ) -> None:
        self.allowed_roots = tuple(root.expanduser().resolve(strict=True) for root in allowed_roots)
        self.capability = capability
        self.environment = environment
        self.development_unsandboxed_override = development_unsandboxed_override
        self._cancellations: dict[str, asyncio.Event] = {}

    async def execute(self, request: ToolWorkerRequest) -> ToolWorkerResponse:
        if not hmac.compare_digest(request.capability, self.capability):
            return self._rejected(request, "authentication_failed")
        try:
            root = self._project_root(request.project_root)
            if request.operation == "cancel":
                target = str(request.args.get("target_request_id", ""))
                cancellation = self._cancellations.get(target)
                if cancellation is not None:
                    cancellation.set()
                return ToolWorkerResponse(
                    request_id=request.request_id,
                    ok=True,
                    returncode=0,
                    status="completed",
                    data={"cancellation_signalled": cancellation is not None},
                )
            if request.operation == "self_check":
                return ToolWorkerResponse(
                    request_id=request.request_id,
                    ok=True,
                    returncode=0,
                    status="completed",
                    data={"self_check": True},
                )
            if request.operation in {"run_command", "test_runner"}:
                return await self._command(request, root)
            if request.operation == "patch_applier":
                return await self._cancellable_mutation(request, root, self._patch)
            if request.operation == "git_commit":
                return await self._cancellable_mutation(request, root, self._git_commit)
            return self._rejected(request, "unknown_operation")
        except (PermissionDeniedError, ValueError, KeyError, TypeError) as exc:
            return self._rejected(request, _safe_validation_code(exc))

    def _project_root(self, value: str) -> Path:
        root = normalize_project_root(value)
        if not any(_relative_to(root, allowed) for allowed in self.allowed_roots):
            raise PermissionDeniedError("Project root is not within a worker allowed root.")
        return root

    async def _command(
        self,
        request: ToolWorkerRequest,
        root: Path,
    ) -> ToolWorkerResponse:
        argv = request.args.get("argv")
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            raise ValueError("invalid_argv")
        command, cwd, _rule = validate_command(argv, root)
        category = (
            ProcessCategory.TEST_RUNNER
            if request.operation == "test_runner"
            else ProcessCategory.RESTRICTED_COMMAND
        )
        profile = (
            ResourceLimitProfile.TEST
            if request.operation == "test_runner"
            else ResourceLimitProfile.COMMAND
        )
        cancellation = asyncio.Event()
        self._cancellations[request.request_id] = cancellation
        try:
            policy = (
                operation_policy(
                    category,
                    project_root=root,
                    allowed_roots=self.allowed_roots,
                )
                if self.environment is not None
                else None
            )
            result = await run_restricted_process(
                command,
                cwd=cwd,
                category=category,
                timeout_seconds=request.timeout_seconds,
                max_stdout_bytes=request.max_stdout_bytes,
                max_stderr_bytes=request.max_stderr_bytes,
                cancellation_event=cancellation,
                resource_limit_profile=profile,
                sandbox_policy=policy,
                sandbox_environment=self.environment or "development",
                development_unsandboxed_override=self.development_unsandboxed_override,
            )
        finally:
            self._cancellations.pop(request.request_id, None)
        return _process_response(request, result)

    async def _patch(
        self,
        request: ToolWorkerRequest,
        root: Path,
        cancellation: asyncio.Event,
    ) -> ToolWorkerResponse:
        encoded = str(request.args["patch_base64"])
        try:
            patch_bytes = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise ValueError("invalid_patch_encoding") from exc
        expected_sha = str(request.args["patch_sha256"])
        expected_length = int(request.args["patch_byte_length"])
        expected_paths = sorted(str(path) for path in request.args["affected_paths"])
        expected_repo = str(request.args["repo_root"])
        expected_state = request.args.get("repo_state_digest")
        if expected_repo != str(root):
            raise PermissionDeniedError("approved_repository_mismatch")
        artifact = await inspect_patch_bytes(
            patch_bytes=patch_bytes,
            repo_root=root,
            sandbox_environment=self.environment,
            development_unsandboxed_override=self.development_unsandboxed_override,
        )
        if artifact.patch_sha256 != expected_sha:
            raise PermissionDeniedError("patch_hash_mismatch")
        if artifact.patch_byte_length != expected_length:
            raise PermissionDeniedError("patch_length_mismatch")
        if artifact.affected_paths != expected_paths:
            raise PermissionDeniedError("patch_paths_mismatch")
        if expected_state is not None and await git_worktree_digest(
            root,
            sandbox_environment=self.environment,
            development_unsandboxed_override=self.development_unsandboxed_override,
        ) != str(expected_state):
            raise PermissionDeniedError("repository_state_changed")
        check_ok, check_stdout, check_stderr = await git_apply_check_bytes(
            root,
            patch_bytes,
            timeout_seconds=request.timeout_seconds,
            cancellation_event=cancellation,
            sandbox_environment=self.environment,
            development_unsandboxed_override=self.development_unsandboxed_override,
        )
        if not check_ok:
            return ToolWorkerResponse(
                request_id=request.request_id,
                ok=False,
                returncode=1,
                stdout=check_stdout[: request.max_stdout_bytes],
                stderr=(check_stderr or "git_apply_check_failed")[: request.max_stderr_bytes],
                status="failed",
                failure_code="git_apply_check_failed",
                data=_artifact_data(artifact),
            )
        code, stdout, stderr = await git_apply_bytes(
            root,
            patch_bytes,
            timeout_seconds=request.timeout_seconds,
            cancellation_event=cancellation,
            sandbox_environment=self.environment,
            development_unsandboxed_override=self.development_unsandboxed_override,
        )
        return ToolWorkerResponse(
            request_id=request.request_id,
            ok=code == 0,
            returncode=code,
            stdout=stdout[: request.max_stdout_bytes],
            stderr=stderr[: request.max_stderr_bytes],
            stdout_truncated=len(stdout) > request.max_stdout_bytes,
            stderr_truncated=len(stderr) > request.max_stderr_bytes,
            status="completed" if code == 0 else "failed",
            failure_code=None if code == 0 else "git_apply_failed",
            data=_artifact_data(artifact),
        )

    async def _git_commit(
        self,
        request: ToolWorkerRequest,
        root: Path,
        cancellation: asyncio.Event,
    ) -> ToolWorkerResponse:
        message = str(request.args["message"])
        if not message or len(message) > 4096 or "\x00" in message:
            raise ValueError("invalid_commit_message")
        expected_digest = str(request.args["staged_diff_sha256"])
        expected_tree = str(request.args["staged_tree_id"])
        if (
            await git_staged_digest(
                root,
                sandbox_environment=self.environment,
                development_unsandboxed_override=self.development_unsandboxed_override,
            )
            != expected_digest
        ):
            raise PermissionDeniedError("staged_diff_changed")
        if (
            await git_staged_tree_id(
                root,
                sandbox_environment=self.environment,
                development_unsandboxed_override=self.development_unsandboxed_override,
            )
            != expected_tree
        ):
            raise PermissionDeniedError("staged_tree_changed")
        code, stdout, stderr = await run_git(
            str(root),
            ["commit", "-m", message],
            timeout=request.timeout_seconds,
            cancellation_event=cancellation,
            sandbox_environment=self.environment,
            development_unsandboxed_override=self.development_unsandboxed_override,
        )
        data: dict[str, Any] = {}
        if code == 0:
            head_code, head, head_error = await run_git(
                str(root),
                ["rev-parse", "HEAD"],
                sandbox_environment=self.environment,
                development_unsandboxed_override=self.development_unsandboxed_override,
            )
            if head_code == 0:
                data["commit_hash"] = head.strip()
            elif head_error:
                stderr = f"{stderr}\n{head_error}".strip()
        return ToolWorkerResponse(
            request_id=request.request_id,
            ok=code == 0,
            returncode=code,
            stdout=stdout[: request.max_stdout_bytes],
            stderr=stderr[: request.max_stderr_bytes],
            stdout_truncated=len(stdout) > request.max_stdout_bytes,
            stderr_truncated=len(stderr) > request.max_stderr_bytes,
            status="completed" if code == 0 else "failed",
            failure_code=None if code == 0 else "git_commit_failed",
            data=data,
        )

    async def _cancellable_mutation(
        self,
        request: ToolWorkerRequest,
        root: Path,
        operation: Any,
    ) -> ToolWorkerResponse:
        cancellation = asyncio.Event()
        self._cancellations[request.request_id] = cancellation
        try:
            return await operation(request, root, cancellation)
        finally:
            self._cancellations.pop(request.request_id, None)

    @staticmethod
    def _rejected(request: ToolWorkerRequest, code: str) -> ToolWorkerResponse:
        return ToolWorkerResponse(
            request_id=request.request_id,
            ok=False,
            status="rejected",
            failure_code=code[:160],
        )


def _process_response(request: ToolWorkerRequest, result: Any) -> ToolWorkerResponse:
    return ToolWorkerResponse(
        request_id=request.request_id,
        ok=result.status is ProcessStatus.COMPLETED and result.returncode == 0,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        stdout_truncated=result.stdout_truncated,
        stderr_truncated=result.stderr_truncated,
        status=result.status.value,
        failure_code=result.failure_code,
        data={
            "resource_limits_applied": list(result.resource_limits.applied),
            "resource_limits_unsupported": list(result.resource_limits.unsupported),
        },
    )


def _artifact_data(artifact: Any) -> dict[str, Any]:
    return {
        "patch_sha256": artifact.patch_sha256,
        "patch_byte_length": artifact.patch_byte_length,
        "affected_paths": artifact.affected_paths,
        "repo_root": artifact.repo_root,
    }


def _relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_validation_code(exc: Exception) -> str:
    value = str(exc)
    known = {
        "invalid_argv",
        "invalid_patch_encoding",
        "invalid_commit_message",
        "approved_repository_mismatch",
        "patch_hash_mismatch",
        "patch_length_mismatch",
        "patch_paths_mismatch",
        "staged_diff_changed",
        "staged_tree_changed",
        "repository_state_changed",
    }
    return value if value in known else "worker_validation_failed"
