from __future__ import annotations

from pathlib import Path

from april_common.audit import AuditLogger
from april_common.settings import AprilSettings

APPROVED_EVOLUTION_TABLES = frozenset(
    {"evolution_runs", "prompt_versions", "playbooks", "playbook_runs", "model_adapters"}
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
        target.write_bytes(data)
        return target

    def write_text(self, path: Path, text: str) -> Path:
        return self.write_bytes(path, text.encode("utf-8"))

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


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
