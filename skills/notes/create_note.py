from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from april_common.settings import get_settings
from april_common.time import utc_now_iso
from skills.base import timed_tool
from skills.schemas import ToolDefinition, ToolResult


def _slug(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "note"


async def create_note(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        settings = get_settings()
        notes_dir = settings.resolve_path(Path("data/notes"))
        notes_dir.mkdir(parents=True, exist_ok=True)
        title = str(args.get("title", "APRIL note"))
        path = notes_dir / f"{_slug(title)}-{uuid.uuid4().hex[:8]}.md"
        content = f"# {title}\n\nCreated: {utc_now_iso()}\n\n{args.get('content', '')}\n"
        path.write_text(content, encoding="utf-8")
        return ToolResult(
            ok=True,
            stdout=str(path),
            data={"path": str(path)},
            risk_level="safe_write",
            permission_level=2,
        )

    return await timed_tool(run, risk_level="safe_write", permission_level=2)


def create_note_definition() -> ToolDefinition:
    return ToolDefinition(
        name="create_note",
        description="Create a local note under APRIL data.",
        permission_level=2,
        risk_level="safe_write",
        allowed_agents={"creative_agent", "general_agent"},
        executor=create_note,
    )
