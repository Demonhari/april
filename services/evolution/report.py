from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from april_common.settings import AprilSettings
from april_common.time import utc_now_iso
from services.evolution.write_guard import EvolutionWriteGuard


def phase_status_summary(phases: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {name: str(payload.get("status", "unknown")) for name, payload in phases.items()}


def build_briefing_summary(
    phases: dict[str, dict[str, Any]], *, pending_eval_cases: int | None = None
) -> str:
    """One-line, briefing-friendly description of what the night changed."""
    parts: list[str] = []
    replay = phases.get("replay", {}).get("counts", {})
    if replay:
        parts.append(f"replayed {sum(replay.values())} run(s)")
    distill = phases.get("distill", {})
    merged = int(distill.get("duplicates_merged", 0) or 0)
    resolved = int(distill.get("contradictions_resolved", 0) or 0)
    fading = int(distill.get("memories_fading", 0) or 0)
    if merged:
        parts.append(f"merged (superseded) {merged} duplicate memorie(s)")
    if resolved:
        parts.append(f"adjudicated {resolved} contradiction(s)")
    if fading:
        parts.append(f"{fading} stale memorie(s) fading")
    mine = phases.get("mine", {})
    mined = mine.get("candidates", [])
    adopted = mine.get("adopted", [])
    if mined:
        parts.append(f"mined {len(mined)} playbook candidate(s)")
    if adopted:
        parts.append(f"auto-adopted {len(adopted)} safe playbook(s)")
    examine = phases.get("examine", {})
    activated = examine.get("activated", [])
    awaiting = examine.get("approval_required", [])
    if activated:
        # Overlay generation is heuristic; activation only means it passed the
        # deterministic eval gate, not that intelligence improved.
        parts.append(f"activated {len(activated)} heuristic prompt overlay(s)")
    if awaiting:
        parts.append(f"{len(awaiting)} overlay(s) await approval")
    if pending_eval_cases:
        parts.append(f"{pending_eval_cases} staged eval case(s) await review")
    skipped = [
        f"{name} ({str(payload.get('reason', 'unknown'))[:80]})"
        for name, payload in phases.items()
        if payload.get("status") == "skipped"
    ]
    if skipped:
        parts.append(f"phase(s) skipped: {', '.join(skipped)}")
    failed = [name for name, payload in phases.items() if payload.get("status") == "failed"]
    if failed:
        parts.append(f"phase(s) failed: {', '.join(failed)}")
    if not parts:
        return "no evolution candidates were produced"
    return "; ".join(parts)


def evolution_report_fields(phases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    evolve = phases.get("evolve", {})
    examine = phases.get("examine", {})
    candidates = evolve.get("candidates", [])
    candidate_count = len(candidates) if isinstance(candidates, list) else 0
    method = str(evolve.get("method", "none" if candidate_count == 0 else "unknown"))
    discarded = examine.get("discarded", [])
    approval_required = examine.get("approval_required", [])
    activated = examine.get("activated", [])
    evaluations = examine.get("evaluations", [])
    if not isinstance(discarded, list):
        discarded = []
    if not isinstance(approval_required, list):
        approval_required = []
    if not isinstance(activated, list):
        activated = []
    if not isinstance(evaluations, list):
        evaluations = []
    below_baseline = [
        item
        for item in discarded
        if isinstance(item, dict) and "baseline" in str(item.get("reason", ""))
    ]
    skipped = [
        {"phase": name, "reason": str(payload.get("reason", "unknown"))[:200]}
        for name, payload in phases.items()
        if payload.get("status") == "skipped"
    ]
    return {
        "candidate_generation": {
            "method": method,
            "deterministic_candidate_count": (
                candidate_count if method == "deterministic-heuristic" else 0
            ),
            "model_generated_candidate_count": (
                candidate_count if method == "model-generated" else 0
            ),
            "candidate_count": candidate_count,
        },
        "candidate_outcomes": {
            "evaluated_count": len(evaluations),
            "activated_count": len(activated),
            "discarded_count": len(discarded),
            "approval_required_count": len(approval_required),
            "below_baseline_count": len(below_baseline),
        },
        "skipped_phases": skipped,
    }


def write_report(
    settings: AprilSettings,
    *,
    guard: EvolutionWriteGuard,
    run_id: str,
    phases: dict[str, dict[str, Any]],
    pending_eval_cases: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    """D6: persist the full nightly report under data/evolution/reports."""
    summary = build_briefing_summary(phases, pending_eval_cases=pending_eval_cases)
    report = {
        "run_id": run_id,
        "status": "completed",
        "reason": summary,
        "summary": summary,
        "phases": phases,
        "phase_statuses": phase_status_summary(phases),
        **evolution_report_fields(phases),
        "pending_eval_cases": pending_eval_cases or 0,
        "created_at": utc_now_iso(),
    }
    path = settings.evolution_path / "reports" / f"{run_id}.json"
    written = guard.write_text(path, json.dumps(report, sort_keys=True, indent=2))
    return written, report
