from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from april_common.settings import AprilSettings
from april_common.time import utc_now_iso
from services.evolution.write_guard import EvolutionWriteGuard


def phase_status_summary(phases: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {name: str(payload.get("status", "unknown")) for name, payload in phases.items()}


def build_briefing_summary(phases: dict[str, dict[str, Any]]) -> str:
    """One-line, briefing-friendly description of what the night changed."""
    parts: list[str] = []
    replay = phases.get("replay", {}).get("counts", {})
    if replay:
        parts.append(f"replayed {sum(replay.values())} run(s)")
    distill = phases.get("distill", {})
    merged = int(distill.get("duplicates_merged", 0) or 0)
    resolved = int(distill.get("contradictions_resolved", 0) or 0)
    if merged:
        parts.append(f"merged {merged} duplicate memorie(s)")
    if resolved:
        parts.append(f"adjudicated {resolved} contradiction(s)")
    mined = phases.get("mine", {}).get("candidates", [])
    if mined:
        parts.append(f"mined {len(mined)} playbook candidate(s)")
    examine = phases.get("examine", {})
    activated = examine.get("activated", [])
    awaiting = examine.get("approval_required", [])
    if activated:
        parts.append(f"activated {len(activated)} prompt overlay(s)")
    if awaiting:
        parts.append(f"{len(awaiting)} overlay(s) await approval")
    failed = [name for name, payload in phases.items() if payload.get("status") == "failed"]
    if failed:
        parts.append(f"phase(s) failed: {', '.join(failed)}")
    if not parts:
        return "no evolution candidates were produced"
    return "; ".join(parts)


def write_report(
    settings: AprilSettings,
    *,
    guard: EvolutionWriteGuard,
    run_id: str,
    phases: dict[str, dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    """D6: persist the full nightly report under data/evolution/reports."""
    summary = build_briefing_summary(phases)
    report = {
        "run_id": run_id,
        "status": "completed",
        "reason": summary,
        "summary": summary,
        "phases": phases,
        "phase_statuses": phase_status_summary(phases),
        "created_at": utc_now_iso(),
    }
    path = settings.evolution_path / "reports" / f"{run_id}.json"
    written = guard.write_text(path, json.dumps(report, sort_keys=True, indent=2))
    return written, report
