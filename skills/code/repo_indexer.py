from __future__ import annotations

import hashlib
from typing import Any

from april_common.path_security import ensure_text_file, normalize_existing_path
from april_common.settings import get_settings
from services.memory.vector_memory import VectorMemory
from skills.base import timed_tool
from skills.filesystem.common import current_path_policy, ignored, read_gitignore_patterns
from skills.schemas import ToolDefinition, ToolResult


def chunk_text(lines: list[str], *, size: int = 80) -> list[tuple[str, int, int]]:
    chunks: list[tuple[str, int, int]] = []
    for index in range(0, len(lines), size):
        chunk_lines = lines[index : index + size]
        chunks.append(("\n".join(chunk_lines), index + 1, index + len(chunk_lines)))
    return chunks


async def repo_indexer(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        settings = get_settings()
        policy = current_path_policy()
        root = normalize_existing_path(args["repo_path"], policy)
        patterns = read_gitignore_patterns(root)
        vector = VectorMemory(settings.vector_index_path)
        indexed = 0
        chunks: list[tuple[str, str, int | None, int | None]] = []
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file() or ignored(file_path, root=root, patterns=patterns):
                continue
            try:
                ensure_text_file(file_path, max_bytes=min(policy.max_read_bytes, 500_000))
            except Exception:
                continue
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            for content, start, end in chunk_text(lines):
                if content.strip():
                    chunks.append((str(file_path), content, start, end))
                    indexed += 1
        source_id = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
        vector.index_chunks(
            source_type="repo",
            source_id=source_id,
            project_id=args.get("project_id"),
            chunks=chunks,
        )
        return ToolResult(
            ok=True,
            stdout=f"Indexed {indexed} chunks.",
            data={"chunks": indexed, "path": str(root)},
            risk_level="safe_write",
            permission_level=2,
        )

    return await timed_tool(run, risk_level="safe_write", permission_level=2)


def repo_indexer_definition() -> ToolDefinition:
    return ToolDefinition(
        name="repo_indexer",
        description="Index a repository into the local vector store.",
        permission_level=2,
        risk_level="safe_write",
        allowed_agents={"coding_agent"},
        executor=repo_indexer,
        affected_paths=lambda args: [str(args.get("repo_path", ""))],
    )
