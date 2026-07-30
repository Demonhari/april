from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from april_common.benchmark_evidence import (
    empty_benchmark_evidence,
    evaluate_benchmark_evidence,
)
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
    empty = empty_benchmark_evidence()
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
    return evaluate_benchmark_evidence(
        report,
        current_hardware_id=safe_hardware_profile()["id"],
        current_config_fingerprint=config_fingerprint_digest(settings.home),
    )
