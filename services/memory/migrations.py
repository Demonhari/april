from __future__ import annotations

from services.memory.database import Database

SCHEMA_VERSION = 10


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
            project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
            actor TEXT NOT NULL DEFAULT 'local-user',
            updated_at TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS conversation_events (
            id TEXT PRIMARY KEY,
            conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
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
            conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
            request_id TEXT,
            intent TEXT,
            agent TEXT,
            model_id TEXT,
            steps_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS reminders (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            due_at TEXT,
            created_at TEXT NOT NULL,
            fired_at TEXT
        );

        CREATE TABLE IF NOT EXISTS scheduler_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS repo_snapshots (
            project_id TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
            last_head_sha TEXT,
            last_dirty_count INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
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
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            completed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS agent_iterations (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
            iteration INTEGER NOT NULL,
            model_id TEXT,
            state TEXT NOT NULL,
            model_output_json TEXT,
            tool_request_json TEXT,
            tool_result_json TEXT,
            approval_id TEXT,
            error TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS suspended_agent_runs (
            id TEXT PRIMARY KEY,
            agent_run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
            approval_id TEXT NOT NULL UNIQUE REFERENCES approvals(id) ON DELETE CASCADE,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
            agent TEXT NOT NULL,
            model_id TEXT,
            iteration INTEGER NOT NULL,
            request_id TEXT NOT NULL,
            messages_json TEXT NOT NULL,
            tool_request_json TEXT NOT NULL,
            normalized_args_json TEXT NOT NULL,
            context_json TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            resumed_at TEXT,
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
    columns = await conn.execute("PRAGMA table_info(conversations)")
    conversation_columns = {row[1] for row in await columns.fetchall()}
    if "project_id" not in conversation_columns:
        await conn.execute("ALTER TABLE conversations ADD COLUMN project_id TEXT")
    if "actor" not in conversation_columns:
        await conn.execute(
            "ALTER TABLE conversations ADD COLUMN actor TEXT NOT NULL DEFAULT 'local-user'"
        )
    if "updated_at" not in conversation_columns:
        await conn.execute("ALTER TABLE conversations ADD COLUMN updated_at TEXT")
        await conn.execute(
            "UPDATE conversations SET updated_at = created_at WHERE updated_at IS NULL"
        )
    columns = await conn.execute("PRAGMA table_info(tasks)")
    task_columns = {row[1] for row in await columns.fetchall()}
    task_column_defs = {
        "conversation_id": "TEXT REFERENCES conversations(id) ON DELETE CASCADE",
        "request_id": "TEXT",
        "intent": "TEXT",
        "agent": "TEXT",
        "model_id": "TEXT",
        "steps_json": "TEXT NOT NULL DEFAULT '[]'",
    }
    for column, definition in task_column_defs.items():
        if column not in task_columns:
            await conn.execute(f"ALTER TABLE tasks ADD COLUMN {column} {definition}")
    columns = await conn.execute("PRAGMA table_info(agent_runs)")
    agent_run_columns = {row[1] for row in await columns.fetchall()}
    if "metadata_json" not in agent_run_columns:
        await conn.execute(
            "ALTER TABLE agent_runs ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
        )
    columns = await conn.execute("PRAGMA table_info(reminders)")
    reminder_columns = {row[1] for row in await columns.fetchall()}
    if "fired_at" not in reminder_columns:
        await conn.execute("ALTER TABLE reminders ADD COLUMN fired_at TEXT")
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
