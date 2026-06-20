from __future__ import annotations

from services.memory.database import Database

SCHEMA_VERSION = 3


async def run_migrations(database: Database) -> None:
    conn = database.connection
    await conn.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
            kind TEXT NOT NULL,
            content TEXT NOT NULL,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            id UNINDEXED,
            content,
            reason,
            tokenize = 'porter'
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tool_calls (
            id TEXT PRIMARY KEY,
            conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
            tool TEXT NOT NULL,
            args_json TEXT NOT NULL,
            result_json TEXT,
            status TEXT NOT NULL,
            permission_level INTEGER NOT NULL,
            risk_level TEXT NOT NULL,
            created_at TEXT NOT NULL,
            completed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS approvals (
            id TEXT PRIMARY KEY,
            tool TEXT NOT NULL,
            args_json TEXT NOT NULL,
            agent TEXT NOT NULL DEFAULT 'general_agent',
            canonical_hash TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            permission_level INTEGER NOT NULL,
            risk_level TEXT NOT NULL,
            status TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            consumed_at TEXT,
            result_json TEXT
        );

        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS reminders (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            due_at TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS repo_indexes (
            id TEXT PRIMARY KEY,
            project_id TEXT REFERENCES projects(id) ON DELETE CASCADE,
            path TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS agent_runs (
            id TEXT PRIMARY KEY,
            conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
            agent TEXT NOT NULL,
            status TEXT NOT NULL,
            model_id TEXT,
            summary TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT
        );
        """
    )
    columns = await conn.execute("PRAGMA table_info(approvals)")
    approval_columns = {row[1] for row in await columns.fetchall()}
    if "agent" not in approval_columns:
        await conn.execute(
            "ALTER TABLE approvals ADD COLUMN agent TEXT NOT NULL DEFAULT 'general_agent'"
        )
    if "metadata_json" not in approval_columns:
        await conn.execute(
            "ALTER TABLE approvals ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
        )
    await conn.execute(
        """
        UPDATE approvals
        SET status = 'expired'
        WHERE status = 'pending'
          AND permission_level >= 3
          AND (metadata_json IS NULL OR metadata_json = '' OR metadata_json = '{}')
        """
    )
    await conn.execute(
        "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES(?, datetime('now'))",
        (SCHEMA_VERSION,),
    )
    await conn.commit()
