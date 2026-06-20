from __future__ import annotations

import asyncio
from typing import Any

from april_common.project_scope import (
    git_apply_check,
    inspect_patch_file,
    normalize_project_child,
    normalize_project_root,
)
from skills.base import timed_tool
from skills.schemas import ToolDefinition, ToolResult


async def patch_applier(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        repo = normalize_project_root(args["repo_path"])
        patch_path = normalize_project_child(
            args["patch_path"],
            project_root=repo,
            must_exist=True,
            allow_absolute=True,
        )
        artifact = await inspect_patch_file(patch_path=args["patch_path"], repo_root=repo)
        check_ok, check_stdout, check_stderr = await git_apply_check(repo, patch_path)
        if not check_ok:
            return ToolResult(
                ok=False,
                stdout=check_stdout,
                stderr=check_stderr or "git apply --check failed.",
                data={
                    "returncode": 1,
                    "patch_sha256": artifact.patch_sha256,
                    "affected_paths": artifact.affected_paths,
                },
                risk_level="code_write",
                permission_level=3,
            )
        process = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(repo),
            "apply",
            "--",
            str(patch_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await process.communicate()
        return ToolResult(
            ok=(process.returncode or 0) == 0,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            data={
                "returncode": process.returncode or 0,
                "patch_sha256": artifact.patch_sha256,
                "affected_paths": artifact.affected_paths,
            },
            risk_level="code_write",
            permission_level=3,
        )

    return await timed_tool(run, risk_level="code_write", permission_level=3)


def patch_applier_definition() -> ToolDefinition:
    return ToolDefinition(
        name="patch_applier",
        description="Apply a patch to a repository after approval.",
        permission_level=3,
        risk_level="code_write",
        confirmation_required=True,
        allowed_agents={"coding_agent"},
        executor=patch_applier,
        affected_paths=lambda args: [
            str(args.get("repo_path", "")),
            str(args.get("patch_path", "")),
        ],
    )
