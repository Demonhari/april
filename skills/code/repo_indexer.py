from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from april_common.audit import AuditLogger
from april_common.path_security import (
    ensure_text_file,
    is_path_within_roots,
    normalize_existing_path,
)
from april_common.settings import get_settings
from april_common.time import utc_now_iso
from services.memory.database import Database
from services.memory.factory import vector_memory_from_settings
from services.memory.repo_index_store import RepoIndexStore
from services.memory.schemas import VectorMetadata
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


def _git_head(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _file_chunks(path_str: str, content: str) -> list[tuple[str, str, int | None, int | None]]:
    return [
        (path_str, body, start, end)
        for body, start, end in chunk_text(content.splitlines())
        if body.strip()
    ]


def _upsert_file_chunks(
    vector: VectorMemory,
    *,
    source_id: str,
    project_id: str | None,
    path_str: str,
    file_chunks: list[tuple[str, str, int | None, int | None]],
) -> None:
    items: list[tuple[str, str, VectorMetadata]] = []
    valid_hashes: set[str] = set()
    for path, body, start, end in file_chunks:
        content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        valid_hashes.add(content_hash)
        metadata = VectorMetadata(
            source_type="repo",
            source_id=source_id,
            project_id=project_id,
            path=path,
            start_line=start,
            end_line=end,
            content_hash=content_hash,
            created_at=utc_now_iso(),
        )
        record_id = hashlib.sha256(f"repo:{source_id}:{path}:{content_hash}".encode()).hexdigest()
        items.append((record_id, body, metadata))
    if items:
        vector.upsert_many(items)
    # Drop earlier chunks for this file whose content changed or disappeared.
    vector.delete_stale_for_path(
        path_str,
        valid_hashes,
        source_type="repo",
        source_id=source_id,
        project_id=project_id,
    )


async def repo_indexer(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        settings = get_settings()
        policy = current_path_policy()
        root = normalize_existing_path(args["repo_path"], policy)
        force = bool(args.get("force_full_reindex", False))
        project_id = args.get("project_id")
        patterns = read_gitignore_patterns(root)
        source_id = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
        git_commit = _git_head(root)
        vector = vector_memory_from_settings(settings, audit=AuditLogger(settings.audit_path))
        max_bytes = min(policy.max_read_bytes, 500_000)

        reindexed_files = 0
        skipped_files = 0
        indexed_chunks = 0
        async with Database(settings.database_path) as database:
            store = RepoIndexStore(database)
            existing = await store.entries_for_source(source_id)
            seen_paths: set[str] = set()
            for file_path in sorted(root.rglob("*")):
                if not file_path.is_file():
                    continue
                # Never follow a link that escapes the repository root.
                if not is_path_within_roots(file_path, [root]):
                    continue
                if ignored(file_path, root=root, patterns=patterns):
                    continue
                try:
                    ensure_text_file(file_path, max_bytes=max_bytes)
                except Exception:
                    continue
                content = file_path.read_text(encoding="utf-8", errors="replace")
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                path_str = str(file_path)
                seen_paths.add(path_str)
                prior = existing.get(path_str)
                if not force and prior is not None and prior.content_hash == content_hash:
                    skipped_files += 1
                    continue
                file_chunks = _file_chunks(path_str, content)
                _upsert_file_chunks(
                    vector,
                    source_id=source_id,
                    project_id=project_id,
                    path_str=path_str,
                    file_chunks=file_chunks,
                )
                indexed_chunks += len(file_chunks)
                await store.upsert_entry(
                    source_id=source_id,
                    project_id=project_id,
                    path=path_str,
                    content_hash=content_hash,
                    git_commit=git_commit,
                    mtime=file_path.stat().st_mtime,
                    indexed_at=utc_now_iso(),
                )
                reindexed_files += 1

            # Files that vanished (deleted or renamed) lose their chunks and rows.
            stale_paths = [path for path in existing if path not in seen_paths]
            for path in stale_paths:
                vector.delete_stale_for_path(
                    path,
                    set(),
                    source_type="repo",
                    source_id=source_id,
                    project_id=project_id,
                )
            removed_files = await store.delete_entries(source_id, stale_paths)

        return ToolResult(
            ok=True,
            stdout=(
                f"Indexed {reindexed_files} file(s) ({indexed_chunks} chunk(s)); "
                f"skipped {skipped_files} unchanged; removed {removed_files} stale."
            ),
            data={
                "path": str(root),
                "source_id": source_id,
                "git_commit": git_commit,
                "reindexed_files": reindexed_files,
                "skipped_files": skipped_files,
                "removed_files": removed_files,
                "chunks": indexed_chunks,
                "forced": force,
            },
            risk_level="safe_write",
            permission_level=2,
        )

    return await timed_tool(run, risk_level="safe_write", permission_level=2)


def repo_indexer_definition() -> ToolDefinition:
    return ToolDefinition(
        name="repo_indexer",
        description="Incrementally index a repository into the local vector store.",
        permission_level=2,
        risk_level="safe_write",
        allowed_agents={"coding_agent"},
        executor=repo_indexer,
        affected_paths=lambda args: [str(args.get("repo_path", ""))],
    )
