from __future__ import annotations

import json
from pathlib import Path

from apps.runner.mac_report import redact_reason
from april_common.settings import AprilSettings


def _pending_write_capable_overlay_count(settings: AprilSettings) -> int:
    """File-only count of stored overlay candidates for write-capable agents.

    Readiness stays inert (no DB connection), so this may include candidates a
    later approval already applied; the check wording says so and points at the
    precise API listing.
    """
    candidates_dir = settings.evolution_path / "candidates"
    if not candidates_dir.is_dir():
        return 0
    write_capable = {"coding_agent", "system_action_agent"}
    count = 0
    for path in candidates_dir.glob("*.overlay.txt"):
        agent = path.name.rsplit("-", 1)[0]
        if agent in write_capable:
            count += 1
    return count


def _pending_eval_case_count(settings: AprilSettings) -> int:
    pending_dir = settings.evolution_path / "evals" / "pending"
    if not pending_dir.is_dir():
        return 0
    return sum(1 for path in pending_dir.glob("*.yaml") if path.is_file())


def _pending_real_runtime_overlay_blockers(settings: AprilSettings) -> list[str]:
    """Redacted reasons from the newest Dreamer report's production real-runtime holdbacks."""
    reports_dir = settings.evolution_path / "reports"
    if not reports_dir.is_dir():
        return []
    newest: tuple[str, float, Path] | None = None
    for path in reports_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            stat = path.stat()
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        created_at = str(payload.get("created_at") or payload.get("generated_at") or "")
        key = (created_at, stat.st_mtime, path)
        if newest is None or key[:2] > newest[:2]:
            newest = key
    if newest is None:
        return []
    try:
        payload = json.loads(newest[2].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    examine = payload.get("phases", {}).get("examine") if isinstance(payload, dict) else None
    if not isinstance(examine, dict):
        return []
    pending = examine.get("pending_real_runtime")
    if not isinstance(pending, list):
        return []
    blockers: list[str] = []
    for item in pending:
        if not isinstance(item, dict):
            continue
        agent = str(item.get("agent") or "unknown")
        status = str(item.get("status") or "unknown")
        reason = redact_reason(str(item.get("reason") or "real-runtime evaluation did not pass"))
        blockers.append(f"{agent}: {status}: {reason}"[:240])
    return blockers
