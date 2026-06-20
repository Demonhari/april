from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from skills.schemas import ToolResult


async def timed_tool(
    func: Callable[[], Awaitable[ToolResult]],
    *,
    risk_level: str,
    permission_level: int,
) -> ToolResult:
    started = time.perf_counter()
    result = await func()
    duration_ms = int((time.perf_counter() - started) * 1000)
    return result.model_copy(
        update={
            "duration_ms": duration_ms,
            "risk_level": result.risk_level or risk_level,
            "permission_level": result.permission_level or permission_level,
        }
    )


def path_args(args: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("path", "repo_path", "directory", "file_path", "target_path"):
        value = args.get(key)
        if isinstance(value, str):
            paths.append(value)
    return paths
