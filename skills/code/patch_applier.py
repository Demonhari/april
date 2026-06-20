from __future__ import annotations

import asyncio
from typing import Any

from april_common.path_security import normalize_existing_path
from skills.base import timed_tool
from skills.filesystem.common import current_path_policy
from skills.schemas import ToolDefinition, ToolResult


async def patch_applier(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        policy = current_path_policy()
        repo = normalize_existing_path(args["repo_path"], policy)
        patch_path = normalize_existing_path(args["patch_path"], policy)
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
            data={"returncode": process.returncode or 0},
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
