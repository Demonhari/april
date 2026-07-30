from __future__ import annotations

import os
import re
import tempfile
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from april_common.audit import AuditLogger
from april_common.settings import AprilSettings
from services.memory.database import Database

APPROVED_EVOLUTION_TABLES = frozenset(
    {
        "evolution_runs",
        "prompt_versions",
        "playbooks",
        "playbook_runs",
        "model_adapters",
        "adapter_operations",
        "evolution_rollouts",
        "evolution_rollout_assignments",
        "evolution_rollout_events",
        # D2 distill/consolidate: duplicate merges and contradiction
        # adjudication update rows (supersede/refresh/resolve) — never delete.
        "memories",
        "memory_contradictions",
    }
)

_MUTATION_TARGET = re.compile(
    r"^\s*(?:INSERT\s+(?:OR\s+\w+\s+)?INTO|UPDATE|DELETE\s+FROM)\s+([A-Za-z_]\w*)",
    re.IGNORECASE,
)


class EvolutionWriteGuard:
    def __init__(self, settings: AprilSettings, *, audit: AuditLogger | None = None) -> None:
        self.settings = settings
        self.audit = audit
        self.allowed_roots = (
            settings.evolution_path.resolve(strict=False),
            settings.playbooks_path.resolve(strict=False),
        )

    def validate_path(self, path: Path) -> Path:
        resolved = path.expanduser().resolve(strict=False)
        if not any(_is_relative_to(resolved, root) for root in self.allowed_roots):
            self._audit_violation("path", str(resolved))
            raise PermissionError("Evolution writes are fenced to runtime evolution/playbook data.")
        return resolved

    def write_bytes(self, path: Path, data: bytes) -> Path:
        target = self.validate_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, raw_temp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
        temp = Path(raw_temp)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, target)
            _fsync_directory(target.parent)
        finally:
            temp.unlink(missing_ok=True)
        return target

    def write_text(self, path: Path, text: str) -> Path:
        return self.write_bytes(path, text.encode("utf-8"))

    def remove_file(self, path: Path) -> None:
        target = self.validate_path(path)
        target.unlink(missing_ok=True)
        _fsync_directory(target.parent)

    def validate_table(self, table: str) -> None:
        if table not in APPROVED_EVOLUTION_TABLES:
            self._audit_violation("table", table)
            raise PermissionError("Evolution database writes are limited to approved tables.")

    def _audit_violation(self, kind: str, target: str) -> None:
        if self.audit is not None:
            self.audit.write(
                {
                    "event_type": "evolution_write_guard_violation",
                    "kind": kind,
                    "target": target,
                    "actor": "dreamer",
                }
            )


class EvolutionDatabaseWriter:
    """Capability-limited mutation facade for evolution-owned tables.

    Callers must name the capability they were given, and the SQL mutation
    target must match it exactly. Read access remains on the normal repository
    interfaces; this object cannot issue arbitrary SELECTs or multi-statement
    scripts.
    """

    def __init__(self, database: Database, guard: EvolutionWriteGuard) -> None:
        self.database = database
        self.guard = guard

    async def execute(self, table: str, sql: str, parameters: Iterable[Any] = ()) -> None:
        self._validate_statement(table, sql)
        await self.database.execute(sql, tuple(parameters))

    @asynccontextmanager
    async def transaction(self, table: str) -> AsyncIterator[_GuardedConnection]:
        self.guard.validate_table(table)
        async with self.database.transaction() as connection:
            yield _GuardedConnection(self, table, connection)

    def _validate_statement(self, table: str, sql: str) -> None:
        self.guard.validate_table(table)
        match = _MUTATION_TARGET.match(sql)
        actual = match.group(1) if match is not None else None
        if actual != table:
            self.guard._audit_violation("sql_table", str(actual or "unparseable"))
            raise PermissionError("Evolution SQL target does not match its table capability.")


class _GuardedConnection:
    def __init__(self, writer: EvolutionDatabaseWriter, table: str, connection: Any) -> None:
        self.writer = writer
        self.table = table
        self.connection = connection

    async def execute(self, sql: str, parameters: Iterable[Any] = ()) -> Any:
        self.writer._validate_statement(self.table, sql)
        return await self.connection.execute(sql, tuple(parameters))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _fsync_directory(path: Path) -> None:
    """Durably publish a replace where the platform supports directory fsync."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
