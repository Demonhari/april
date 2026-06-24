from __future__ import annotations

import hashlib
from dataclasses import dataclass

from services.memory.database import Database


@dataclass(frozen=True, slots=True)
class RepoIndexEntry:
    path: str
    content_hash: str
    git_commit: str | None
    mtime: float | None
    indexed_at: str


def _entry_id(source_id: str, path: str) -> str:
    # One row per (repo source, file path); also the deletion key for stale chunks.
    return hashlib.sha256(f"{source_id}:{path}".encode()).hexdigest()


class RepoIndexStore:
    """Per-file repo index metadata in the ``repo_indexes`` table.

    This is the source of truth that drives safe incremental indexing: only
    changed or new files are re-embedded, and rows for deleted/renamed files are
    removed along with their chunks.
    """

    def __init__(self, database: Database) -> None:
        self.database = database

    async def entries_for_source(self, source_id: str) -> dict[str, RepoIndexEntry]:
        rows = await self.database.fetchall(
            "SELECT path, content_hash, git_commit, mtime, indexed_at "
            "FROM repo_indexes WHERE source_id = ?",
            (source_id,),
        )
        return {
            row["path"]: RepoIndexEntry(
                path=row["path"],
                content_hash=row["content_hash"],
                git_commit=row["git_commit"],
                mtime=row["mtime"],
                indexed_at=row["indexed_at"],
            )
            for row in rows
        }

    async def upsert_entry(
        self,
        *,
        source_id: str,
        project_id: str | None,
        path: str,
        content_hash: str,
        git_commit: str | None,
        mtime: float | None,
        indexed_at: str,
    ) -> None:
        await self.database.execute(
            """
            INSERT INTO repo_indexes(
                id, project_id, source_id, path, content_hash, git_commit, mtime,
                indexed_at, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                project_id = excluded.project_id,
                content_hash = excluded.content_hash,
                git_commit = excluded.git_commit,
                mtime = excluded.mtime,
                indexed_at = excluded.indexed_at
            """,
            (
                _entry_id(source_id, path),
                project_id,
                source_id,
                path,
                content_hash,
                git_commit,
                mtime,
                indexed_at,
                indexed_at,
            ),
        )

    async def delete_entries(self, source_id: str, paths: list[str]) -> int:
        removed = 0
        for path in paths:
            cursor = await self.database.execute(
                "DELETE FROM repo_indexes WHERE id = ?", (_entry_id(source_id, path),)
            )
            removed += int(cursor.rowcount or 0)
        return removed
