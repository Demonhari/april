from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from april_common.config_fingerprint import config_fingerprint_digest
from april_common.hardware_profile import safe_hardware_profile
from april_common.settings import AprilSettings
from april_common.verification_evidence import verified_model_ids


def _verified_model_ids(home: Path) -> set[str]:
    return verified_model_ids(home)


def _active_vector_provider(path: Path) -> str | None:
    metadata = _active_vector_metadata(path)
    provider = metadata.get("provider")
    return str(provider) if isinstance(provider, str) else None


def _active_vector_metadata(path: Path) -> dict[str, object]:
    try:
        generation_id = (path / "CURRENT").read_text(encoding="utf-8").strip()
        if not generation_id or "/" in generation_id or "\\" in generation_id:
            return {}
        metadata = json.loads(
            (path / "generations" / generation_id / "metadata.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(metadata, dict):
        return {}
    return {**metadata, "active_generation": generation_id}


def _benchmark_evidence(settings: AprilSettings) -> dict[str, object]:
    empty: dict[str, object] = {
        "exists": False,
        "current_hardware": False,
        "simulated": False,
        "stale": False,
        "incomplete": False,
        "production_eligible": False,
    }
    if not settings.database_path.is_file():
        return empty
    try:
        connection = sqlite3.connect(f"file:{settings.database_path}?mode=ro", uri=True)
        row = connection.execute(
            """
            SELECT result_json FROM background_jobs
            WHERE job_type = 'model_setup_comparison' AND status = 'succeeded'
            ORDER BY completed_at DESC LIMIT 1
            """
        ).fetchone()
    except (sqlite3.Error, OSError):
        return empty
    finally:
        if "connection" in locals():
            connection.close()
    if row is None or not isinstance(row[0], str):
        return empty
    try:
        report = json.loads(row[0])
    except json.JSONDecodeError:
        return {**empty, "incomplete": True}
    if not isinstance(report, dict):
        return {**empty, "incomplete": True}
    profile = report.get("hardware_profile")
    current = safe_hardware_profile()["id"]
    profile_id = profile.get("id") if isinstance(profile, dict) else None
    simulated = bool(report.get("simulated"))
    unavailable = report.get("unavailable_measurements")
    required_unavailable = (
        [item for item in unavailable if item != "thermal_throttling"]
        if isinstance(unavailable, list)
        else []
    )
    report_fingerprint = report.get("config_fingerprint")
    fingerprint_matches = isinstance(
        report_fingerprint, str
    ) and report_fingerprint == config_fingerprint_digest(settings.home)
    incomplete = (
        bool(required_unavailable) or not bool(report.get("fixture_set")) or not fingerprint_matches
    )
    current_hardware = profile_id == current
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
