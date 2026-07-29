from __future__ import annotations

from services.memory.database import Database

SCHEMA_VERSION = 20


async def run_migrations(database: Database) -> None:
    async with database.write_coordination():
        await _run_migrations_locked(database)


async def _run_migrations_locked(database: Database) -> None:
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
            created_at TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.7,
            source TEXT NOT NULL DEFAULT 'user',
            last_used_at TEXT,
            use_count INTEGER NOT NULL DEFAULT 0,
            expires_at TEXT,
            superseded_by TEXT REFERENCES memories(id),
            content_encrypted INTEGER NOT NULL DEFAULT 0
                CHECK (content_encrypted IN (0, 1))
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            id UNINDEXED,
            content,
            reason,
            tokenize = "unicode61 remove_diacritics 0 tokenchars '_'"
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

        CREATE TABLE IF NOT EXISTS conversation_summaries (
            conversation_id TEXT PRIMARY KEY
                REFERENCES conversations(id) ON DELETE CASCADE,
            summary_json TEXT NOT NULL,
            through_message_id TEXT NOT NULL,
            through_created_at TEXT NOT NULL,
            summarized_message_count INTEGER NOT NULL,
            source_hash TEXT NOT NULL,
            model_id TEXT,
            version INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
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

        CREATE TABLE IF NOT EXISTS background_jobs (
            id TEXT PRIMARY KEY,
            job_type TEXT NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN (
                    'queued', 'running', 'cancelling', 'cancelled',
                    'succeeded', 'failed', 'interrupted'
                )
            ),
            payload_json TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            owner TEXT NOT NULL,
            conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
            project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
            progress_percent INTEGER NOT NULL DEFAULT 0
                CHECK (progress_percent BETWEEN 0 AND 100),
            progress_code TEXT,
            attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
            maximum_attempts INTEGER NOT NULL DEFAULT 1 CHECK (maximum_attempts >= 1),
            cancellation_requested INTEGER NOT NULL DEFAULT 0
                CHECK (cancellation_requested IN (0, 1)),
            worker_id TEXT,
            lease_acquired_at TEXT,
            lease_expires_at TEXT,
            heartbeat_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            result_json TEXT,
            error_code TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_background_jobs_claim
            ON background_jobs(status, lease_expires_at, created_at);
        CREATE INDEX IF NOT EXISTS idx_background_jobs_project
            ON background_jobs(project_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_background_jobs_owner
            ON background_jobs(owner, created_at);

        CREATE TABLE IF NOT EXISTS background_job_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL REFERENCES background_jobs(id) ON DELETE CASCADE,
            event_type TEXT NOT NULL,
            message_code TEXT,
            progress_percent INTEGER CHECK (
                progress_percent IS NULL OR progress_percent BETWEEN 0 AND 100
            ),
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_background_job_events_job
            ON background_job_events(job_id, id);

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

        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
            source TEXT NOT NULL,
            started_at TEXT NOT NULL,
            last_activity_at TEXT NOT NULL,
            closed_at TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS wake_events (
            id TEXT PRIMARY KEY,
            session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
            source TEXT NOT NULL,
            score REAL,
            accepted INTEGER NOT NULL DEFAULT 1,
            reason TEXT,
            transcript_present INTEGER NOT NULL DEFAULT 0,
            captured_at TEXT,
            session_hint TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS memory_contradictions (
            id TEXT PRIMARY KEY,
            memory_id_a TEXT NOT NULL REFERENCES memories(id),
            memory_id_b TEXT NOT NULL REFERENCES memories(id),
            status TEXT NOT NULL DEFAULT 'pending',
            resolution TEXT,
            created_at TEXT NOT NULL,
            resolved_at TEXT
        );

        CREATE TABLE IF NOT EXISTS feedback_events (
            id TEXT PRIMARY KEY,
            session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
            conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
            agent_run_id TEXT REFERENCES agent_runs(id) ON DELETE SET NULL,
            rating TEXT NOT NULL,
            reason TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS playbooks (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            source TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'candidate',
            trigger_examples_json TEXT NOT NULL DEFAULT '[]',
            steps_json TEXT NOT NULL DEFAULT '[]',
            required_permission_level INTEGER NOT NULL DEFAULT 1,
            stats_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS playbook_runs (
            id TEXT PRIMARY KEY,
            playbook_id TEXT NOT NULL REFERENCES playbooks(id) ON DELETE CASCADE,
            conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
            status TEXT NOT NULL,
            steps_completed INTEGER NOT NULL DEFAULT 0,
            detail TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS memory_index_repairs (
            memory_id TEXT PRIMARY KEY,
            operation TEXT NOT NULL,
            error_type TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS memory_provenance (
            memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
            source_session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
            source_conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
            source_message_ids_json TEXT NOT NULL DEFAULT '[]',
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS evolution_runs (
            id TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            status TEXT NOT NULL,
            phases_json TEXT NOT NULL DEFAULT '{}',
            report_path TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS prompt_versions (
            id TEXT PRIMARY KEY,
            agent TEXT NOT NULL,
            version INTEGER NOT NULL,
            overlay_path TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 0,
            eval_score REAL,
            baseline_score REAL,
            created_at TEXT NOT NULL,
            UNIQUE(agent, version)
        );

        CREATE TABLE IF NOT EXISTS model_adapters (
            id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            adapter_path TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'inactive',
            eval_score REAL,
            baseline_score REAL,
            created_at TEXT NOT NULL,
            activated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS adapter_operations (
            id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            operation_type TEXT NOT NULL,
            status TEXT NOT NULL,
            previous_active_version INTEGER,
            requested_target_version INTEGER NOT NULL,
            previous_pointer_json TEXT,
            target_pointer_json TEXT NOT NULL,
            target_adapter_path TEXT NOT NULL,
            target_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            error_code TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_adapter_operations_status
        ON adapter_operations(status, model_id, created_at);

        CREATE TABLE IF NOT EXISTS routing_outcomes (
            id TEXT PRIMARY KEY,
            agent_run_id TEXT REFERENCES agent_runs(id) ON DELETE SET NULL,
            route_key TEXT NOT NULL,
            intent TEXT NOT NULL,
            agent TEXT NOT NULL,
            route_source TEXT NOT NULL,
            normalized_tool_class TEXT NOT NULL,
            raw_confidence REAL,
            effective_confidence REAL NOT NULL,
            structured_output_valid INTEGER NOT NULL DEFAULT 1,
            repair_used INTEGER NOT NULL DEFAULT 0,
            tool_outcome TEXT,
            approval_outcome TEXT,
            user_correction INTEGER NOT NULL DEFAULT 0,
            negative_feedback INTEGER NOT NULL DEFAULT 0,
            regeneration_or_retry INTEGER NOT NULL DEFAULT 0,
            coding_test_outcome TEXT,
            final_status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_routing_outcomes_route_created
        ON routing_outcomes(route_key, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_routing_outcomes_agent_run
        ON routing_outcomes(agent_run_id);
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
    columns = await conn.execute("PRAGMA table_info(users)")
    user_columns = {row[1] for row in await columns.fetchall()}
    user_column_defs = {
        "address": "TEXT",
        "timezone": "TEXT",
        "preferences_json": "TEXT NOT NULL DEFAULT '{}'",
        "updated_at": "TEXT",
    }
    for column, definition in user_column_defs.items():
        if column not in user_columns:
            await conn.execute(f"ALTER TABLE users ADD COLUMN {column} {definition}")
    columns = await conn.execute("PRAGMA table_info(memories)")
    memory_columns = {row[1] for row in await columns.fetchall()}
    memory_column_defs = {
        "confidence": "REAL NOT NULL DEFAULT 0.7",
        "source": "TEXT NOT NULL DEFAULT 'user'",
        "last_used_at": "TEXT",
        "use_count": "INTEGER NOT NULL DEFAULT 0",
        "expires_at": "TEXT",
        "superseded_by": "TEXT REFERENCES memories(id)",
        "content_encrypted": "INTEGER NOT NULL DEFAULT 0",
    }
    for column, definition in memory_column_defs.items():
        if column not in memory_columns:
            await conn.execute(f"ALTER TABLE memories ADD COLUMN {column} {definition}")
    columns = await conn.execute("PRAGMA table_info(wake_events)")
    wake_event_columns = {row[1] for row in await columns.fetchall()}
    for column in ("captured_at", "session_hint"):
        if column not in wake_event_columns:
            await conn.execute(f"ALTER TABLE wake_events ADD COLUMN {column} TEXT")
    columns = await conn.execute("PRAGMA table_info(repo_indexes)")
    repo_index_columns = {row[1] for row in await columns.fetchall()}
    repo_index_column_defs = {
        "source_id": "TEXT",
        "git_commit": "TEXT",
        "mtime": "REAL",
        "indexed_at": "TEXT",
    }
    for column, definition in repo_index_column_defs.items():
        if column not in repo_index_columns:
            await conn.execute(f"ALTER TABLE repo_indexes ADD COLUMN {column} {definition}")
    columns = await conn.execute("PRAGMA table_info(playbook_runs)")
    playbook_run_columns = {row[1] for row in await columns.fetchall()}
    playbook_run_column_defs = {
        "current_step_index": "INTEGER NOT NULL DEFAULT 0",
        "expanded_steps_json": "TEXT NOT NULL DEFAULT '[]'",
        "snapshot_hash": "TEXT",
        "step_states_json": "TEXT NOT NULL DEFAULT '[]'",
        "pending_approval_id": "TEXT",
        "pending_action_hash": "TEXT",
        "updated_at": "TEXT",
        "error_json": "TEXT",
        "duration_ms": "REAL",
        "agent_id": "TEXT NOT NULL DEFAULT 'general_agent'",
    }
    for column, definition in playbook_run_column_defs.items():
        if column not in playbook_run_columns:
            await conn.execute(f"ALTER TABLE playbook_runs ADD COLUMN {column} {definition}")
    await conn.execute("UPDATE playbook_runs SET updated_at = created_at WHERE updated_at IS NULL")
    # Pre-v14 pending runs have no immutable expanded-step snapshot and cannot
    # be resumed safely. Close them explicitly instead of guessing what to run.
    await conn.execute(
        """
        UPDATE playbook_runs
        SET status = 'failed',
            detail = COALESCE(detail, 'Legacy pending run cannot be resumed safely.'),
            completed_at = COALESCE(completed_at, datetime('now')),
            updated_at = datetime('now')
        WHERE status = 'pending_approval'
          AND (snapshot_hash IS NULL OR expanded_steps_json = '[]')
        """
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
    await conn.commit()
    await _migrate_memories_fts_v18(database)
    await conn.execute(
        "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES(?, datetime('now'))",
        (SCHEMA_VERSION,),
    )
    await conn.commit()


async def _migrate_memories_fts_v18(database: Database) -> None:
    """Atomically replace the legacy Porter FTS index with Unicode61.

    ``memories`` remains authoritative. The replacement is populated before the
    old virtual table is dropped, and the complete swap is one SQLite
    transaction. A leftover temporary table from an interrupted attempt is
    harmless and is rebuilt on the next startup.
    """
    conn = database.connection
    definition_cursor = await conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'memories_fts'"
    )
    definition_row = await definition_cursor.fetchone()
    definition = str(definition_row[0] or "") if definition_row is not None else ""
    memory_count_cursor = await conn.execute("SELECT COUNT(*) FROM memories")
    memory_count_row = await memory_count_cursor.fetchone()
    if memory_count_row is None:
        raise RuntimeError("Unable to count authoritative memory rows during migration.")
    memory_count = int(memory_count_row[0])
    fts_count = -1
    if definition:
        try:
            fts_count_cursor = await conn.execute("SELECT COUNT(*) FROM memories_fts")
            fts_count_row = await fts_count_cursor.fetchone()
            fts_count = int(fts_count_row[0]) if fts_count_row is not None else -1
        except Exception:
            fts_count = -1
    if "unicode61" in definition.casefold() and fts_count == memory_count:
        return

    try:
        await conn.execute("BEGIN IMMEDIATE")
        await conn.execute("DROP TABLE IF EXISTS memories_fts_v18")
        await conn.execute(
            """
            CREATE VIRTUAL TABLE memories_fts_v18 USING fts5(
                id UNINDEXED,
                content,
                reason,
                tokenize = "unicode61 remove_diacritics 0 tokenchars '_'"
            )
            """
        )
        await conn.execute(
            """
            INSERT INTO memories_fts_v18(id, content, reason)
            SELECT id, content, reason FROM memories ORDER BY rowid
            """
        )
        await conn.execute("DROP TABLE IF EXISTS memories_fts")
        await conn.execute("ALTER TABLE memories_fts_v18 RENAME TO memories_fts")
        await conn.commit()
    except Exception:
        await conn.rollback()
        raise
