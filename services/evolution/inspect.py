from __future__ import annotations

import difflib
from datetime import datetime
from pathlib import Path
from typing import Any

from april_common.settings import AprilSettings
from april_common.time import utc_now
from services.evolution.dreamer import _report_sort_key, latest_report
from services.evolution.feedback_eval import count_pending_eval_cases
from services.evolution.scheduler import (
    _LAST_EVOLUTION_DATE_KEY,
    _inside_window,
    evolution_kill_switch_active,
    evolution_kill_switch_path,
)
from services.evolution.write_guard import EvolutionWriteGuard
from services.memory.database import Database

_DIFF_MAX_CHARS = 20_000

# Agents whose overlays never auto-apply; mirrored from versions.py so a
# file-only count needs no database connection.
_WRITE_CAPABLE_AGENTS = frozenset({"coding_agent", "system_action_agent"})


def latest_report_basename(settings: AprilSettings) -> str | None:
    """Basename (never a path) of the newest Dreamer report, if any."""
    reports = list((settings.evolution_path / "reports").glob("*.json"))
    if not reports:
        return None
    return max(reports, key=_report_sort_key).name


def count_pending_write_capable_overlay_candidates(settings: AprilSettings) -> int:
    """File-only count of stored overlay candidates for write-capable agents.

    This may include candidates a later approval already applied; the precise
    pending list is served by ``PromptOverlayApprovalService.list_pending``.
    """
    candidates_dir = settings.evolution_path / "candidates"
    if not candidates_dir.is_dir():
        return 0
    return sum(
        1
        for path in candidates_dir.glob("*.overlay.txt")
        if path.name.rsplit("-", 1)[0] in _WRITE_CAPABLE_AGENTS
    )


async def scheduler_gate_reason(
    settings: AprilSettings,
    database: Database,
    *,
    now: datetime | None = None,
) -> str | None:
    """Why the Dreamer would not run right now, or ``None`` if it could.

    Mirrors :class:`EvolutionSchedulerGate` minus the resource-governor sample
    (CPU/power probing is too heavy for a health endpoint). Reasons are fixed
    safe strings — never a path or transcript.
    """
    current = now or utc_now()
    if evolution_kill_switch_active(settings):
        return "disabled by local kill switch"
    if not settings.evolution.enabled:
        return "evolution disabled"
    if not settings.scheduler.enabled:
        return "scheduler disabled"
    if not _inside_window(current.time(), settings.evolution.window):
        return "outside evolution window"
    row = await database.fetchone(
        "SELECT value FROM scheduler_state WHERE key = ?",
        (_LAST_EVOLUTION_DATE_KEY,),
    )
    if row is not None and str(row["value"]) == current.date().isoformat():
        return "already ran today"
    return None


async def evolution_health_snapshot(settings: AprilSettings, database: Database) -> dict[str, Any]:
    """Redacted evolution block for the unauthenticated ``/health`` endpoint.

    Booleans, counts, dates, and fixed reason strings only — never a path,
    report body, or transcript content.
    """
    last_run_row = await database.fetchone(
        "SELECT date FROM evolution_runs ORDER BY created_at DESC LIMIT 1"
    )
    return {
        "enabled": settings.evolution.enabled,
        "kill_switch_active": evolution_kill_switch_active(settings),
        "scheduler_enabled": settings.scheduler.enabled,
        "dreamer_last_run_date": (str(last_run_row["date"]) if last_run_row is not None else None),
        "dreamer_last_report_available": latest_report_basename(settings) is not None,
        "pending_eval_case_count": count_pending_eval_cases(settings),
        "pending_write_capable_overlay_count": (
            count_pending_write_capable_overlay_candidates(settings)
        ),
        "last_skip_reason": await scheduler_gate_reason(settings, database),
    }


async def evolution_status(settings: AprilSettings, database: Database) -> dict[str, Any]:
    """Read-only snapshot of the self-evolution subsystem."""
    last_run_row = await database.fetchone(
        "SELECT id, date, status, created_at, completed_at FROM evolution_runs "
        "ORDER BY created_at DESC LIMIT 1"
    )
    overlay_counts = await database.fetchall(
        "SELECT agent, COUNT(*) AS versions, MAX(active) AS has_active "
        "FROM prompt_versions GROUP BY agent ORDER BY agent"
    )
    report = latest_report(settings)
    report_summary: dict[str, Any] | None = None
    if report is not None:
        report_summary = {
            "run_id": report.get("run_id"),
            "created_at": report.get("created_at"),
            "summary": report.get("summary"),
            "phase_statuses": report.get("phase_statuses"),
        }
    return {
        "enabled": settings.evolution.enabled,
        "kill_switch_active": evolution_kill_switch_active(settings),
        "scheduler_enabled": settings.scheduler.enabled,
        "window": settings.evolution.window,
        "require_ac_power": settings.evolution.require_ac_power,
        "max_minutes": settings.evolution.max_minutes,
        "daily_memory_cap": settings.evolution.daily_memory_cap,
        "last_run": dict(last_run_row) if last_run_row is not None else None,
        "last_report_basename": latest_report_basename(settings),
        "overlays": [dict(row) for row in overlay_counts],
        "pending_write_capable_overlay_count": (
            count_pending_write_capable_overlay_candidates(settings)
        ),
        "pending_eval_case_count": count_pending_eval_cases(settings),
        "current_gate_reason": await scheduler_gate_reason(settings, database),
        "latest_report": report_summary,
    }


async def evolution_history(database: Database, *, limit: int = 20) -> list[dict[str, Any]]:
    capped = max(1, min(limit, 200))
    rows = await database.fetchall(
        "SELECT id, date, status, phases_json, created_at, completed_at "
        "FROM evolution_runs ORDER BY created_at DESC LIMIT ?",
        (capped,),
    )
    return [dict(row) for row in rows]


async def overlay_diff(
    settings: AprilSettings,
    database: Database,
    *,
    agent: str,
    from_version: int | None = None,
    to_version: int | None = None,
) -> dict[str, Any]:
    """Unified diff between two overlay versions of one agent.

    Defaults compare the previous version against the newest one. Overlay
    files that were removed on disk (data/evolution deleted) diff as empty.
    """
    rows = await database.fetchall(
        "SELECT version, overlay_path, active FROM prompt_versions "
        "WHERE agent = ? ORDER BY version",
        (agent,),
    )
    versions = [int(row["version"]) for row in rows]
    if not versions:
        return {"agent": agent, "error": "no overlay versions for agent", "diff": None}
    resolved_to = to_version if to_version is not None else versions[-1]
    if from_version is not None:
        resolved_from = from_version
    else:
        earlier = [version for version in versions if version < resolved_to]
        resolved_from = earlier[-1] if earlier else resolved_to
    paths = {int(row["version"]): Path(str(row["overlay_path"])) for row in rows}
    if resolved_from not in paths or resolved_to not in paths:
        return {"agent": agent, "error": "version not found", "diff": None}
    from_text = _read_or_empty(paths[resolved_from])
    to_text = _read_or_empty(paths[resolved_to])
    diff = "\n".join(
        difflib.unified_diff(
            from_text.splitlines(),
            to_text.splitlines(),
            fromfile=f"{agent}/v{resolved_from}",
            tofile=f"{agent}/v{resolved_to}",
            lineterm="",
        )
    )
    return {
        "agent": agent,
        "from_version": resolved_from,
        "to_version": resolved_to,
        "diff": diff[:_DIFF_MAX_CHARS],
        "truncated": len(diff) > _DIFF_MAX_CHARS,
    }


def set_evolution_kill_switch(settings: AprilSettings, *, disabled: bool) -> dict[str, Any]:
    """Flip the local evolution kill switch (a flag file inside the fence)."""
    path = evolution_kill_switch_path(settings)
    if disabled:
        EvolutionWriteGuard(settings).write_text(
            path, "Dreamer disabled by local operator via `april evolve off`.\n"
        )
    else:
        path.unlink(missing_ok=True)
    return {
        "kill_switch_active": evolution_kill_switch_active(settings),
        "evolution_enabled_in_config": settings.evolution.enabled,
    }


def _read_or_empty(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""
