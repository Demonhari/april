from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ACTIVITY_MAX_LIMIT = 200
ACTIVITY_ALLOWED_KEYS = frozenset(
    {
        "timestamp",
        "event_type",
        "event",
        "actor",
        "request_id",
        "audit_correlation_id",
        "approval_id",
        "reference_id",
        "reminder_id",
        "memory_id",
        "memory_type",
        "agent",
        "tool",
        "permission_level",
        "risk",
        "risk_level",
        "outcome",
        "status",
        "project_id",
        "content_length",
        "reason_length",
        "kind",
        "sink",
        "date",
        "muted",
        "case_id",
    }
)


def read_activity_events(audit_path: Path, limit: int) -> list[dict[str, Any]]:
    if not audit_path.exists():
        return []
    try:
        lines = audit_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        safe_source = dict(record)
        payload = record.get("payload")
        if isinstance(payload, dict):
            safe_source.update(payload)
        projected = {
            key: value for key, value in safe_source.items() if key in ACTIVITY_ALLOWED_KEYS
        }
        if projected:
            events.append(projected)
        if len(events) >= limit:
            break
    return events
