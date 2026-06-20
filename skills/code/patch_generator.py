from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from april_common.settings import get_settings
from skills.base import timed_tool
from skills.schemas import ToolDefinition, ToolResult


async def patch_generator(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        settings = get_settings()
        patch_dir = settings.resolve_path(Path("data/patches"))
        patch_dir.mkdir(parents=True, exist_ok=True)
        patch_path = patch_dir / f"{uuid.uuid4()}.patch"
        content = str(args.get("patch", ""))
        patch_path.write_text(content, encoding="utf-8")
        return ToolResult(
            ok=True,
            stdout=str(patch_path),
            data={"patch_path": str(patch_path)},
            risk_level="safe_write",
            permission_level=2,
        )

    return await timed_tool(run, risk_level="safe_write", permission_level=2)


def patch_generator_definition() -> ToolDefinition:
    return ToolDefinition(
        name="patch_generator",
        description="Write a draft patch file without applying it.",
        permission_level=2,
        risk_level="safe_write",
        allowed_agents={"coding_agent"},
        executor=patch_generator,
        affected_paths=lambda args: [],
    )
