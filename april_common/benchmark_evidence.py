from __future__ import annotations


def empty_benchmark_evidence() -> dict[str, object]:
    return {
        "exists": False,
        "current_hardware": False,
        "simulated": False,
        "stale": False,
        "incomplete": False,
        "production_eligible": False,
    }


def evaluate_benchmark_evidence(
    report: object,
    *,
    current_hardware_id: str,
    current_config_fingerprint: str | None,
) -> dict[str, object]:
    """Classify redacted comparison evidence without treating absence as success."""
    empty = empty_benchmark_evidence()
    if not isinstance(report, dict):
        return {**empty, "incomplete": report is not None}
    profile = report.get("hardware_profile")
    profile_id = profile.get("id") if isinstance(profile, dict) else None
    simulated = bool(report.get("simulated"))
    unavailable = report.get("unavailable_measurements")
    required_unavailable = (
        [item for item in unavailable if item != "thermal_throttling"]
        if isinstance(unavailable, list)
        else []
    )
    report_fingerprint = report.get("config_fingerprint")
    fingerprint_matches = (
        isinstance(report_fingerprint, str) and report_fingerprint == current_config_fingerprint
    )
    incomplete = (
        bool(required_unavailable) or not bool(report.get("fixture_set")) or not fingerprint_matches
    )
    current_hardware = profile_id == current_hardware_id
    return {
        "exists": not simulated,
        "current_hardware": current_hardware,
        "simulated": simulated,
        "stale": (bool(profile_id) and not current_hardware) or not fingerprint_matches,
        "incomplete": incomplete,
        "production_eligible": bool(report.get("production_eligible"))
        and current_hardware
        and not simulated
        and not incomplete,
    }
