from __future__ import annotations

# ruff: noqa: F401
# mypy: disable-error-code="attr-defined"
import asyncio
import hashlib
import json
import re
import sqlite3
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

import aiosqlite

from april_common.audit import AuditLogger
from april_common.settings import AprilSettings
from april_common.time import parse_utc_iso, utc_now, utc_now_iso
from services.evolution.eval_review import list_reviewed_eval_cases
from services.evolution.evaluator import RuntimeEvalClient, evaluate_overlay_candidate_real_runtime
from services.evolution.rollout_models import (
    _IDENTIFIER_RE,
    _SAFE_OUTCOME_KEYS,
    _SHA256_RE,
    _TRANSITIONS,
    TERMINAL_STATES,
    CanaryContext,
    CanarySelection,
    CandidateType,
    FaultHook,
    InvalidRolloutTransition,
    PromotionReadiness,
    RolloutBlocked,
    RolloutError,
    RolloutRecord,
    RolloutState,
    ShadowEvaluator,
    ShadowMetrics,
)
from services.evolution.rollout_policy import (
    _aggregate_outcome,
    _canary_eligible,
    _canonical_json,
    _encode_column_value,
    _outcome_event_summary,
    _reason_code,
    _sha256_file,
    _sha256_text,
    _validate_identifier,
    _validate_safe_outcome,
    _validate_sha256,
)
from services.evolution.rollout_records import _record_from_row
from services.evolution.versions import prompt_overlay_rejection_reason
from services.evolution.write_guard import EvolutionWriteGuard
from services.memory.database import Database
from services.permissions.approvals import ApprovalStore, canonical_hash
from services.permissions.schemas import ApprovalRequest


def inspect_rollout_state(settings: AprilSettings) -> dict[str, Any]:
    """Read-only, redaction-safe rollout probe for offline readiness."""

    enabled = bool(settings.evolution.enabled and settings.evolution.rollout_enabled)
    base = {
        "enabled": enabled,
        "canary_enabled": settings.evolution.canary_enabled,
        "automatic_candidate_creation": settings.evolution.automatic_candidate_creation,
        "automatic_promotion": settings.evolution.automatic_promotion,
        "lora_canary_supported": False,
        "lora_canary_readiness_reason": "runtime_candidate_capability_unavailable",
        "baseline_model_instance": None,
        "candidate_instances": [],
        "candidate_integrity_state": "unknown",
        "rollout_state": "inactive",
        "rollback_required": False,
        "incomplete_transition_count": 0,
        "active_canary_count": 0,
        "expired_canary_count": 0,
        "candidate_unavailable_count": 0,
        "candidate_hash_mismatch_count": 0,
        "pointer_database_disagreement_count": 0,
        "rollback_required_count": 0,
    }
    database_path = settings.database_path.expanduser().resolve(strict=False)
    if not database_path.is_file():
        return {**base, "status": "disabled" if not enabled else "not_initialized"}
    connection = sqlite3.connect(f"{database_path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        table = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'evolution_rollouts'
            """
        ).fetchone()
        if table is None:
            return {**base, "status": "disabled" if not enabled else "not_initialized"}
        rows = list(connection.execute("SELECT * FROM evolution_rollouts"))
        runtime_table = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'evolution_rollout_runtime'
            """
        ).fetchone()
        runtime_rows = (
            list(
                connection.execute(
                    "SELECT * FROM evolution_rollout_runtime ORDER BY updated_at DESC"
                )
            )
            if runtime_table is not None
            else []
        )
        active_prompt = {
            str(row["agent"]): str(row["content_hash"])
            for row in connection.execute(
                "SELECT agent, content_hash FROM prompt_versions WHERE active = 1"
            )
        }
    finally:
        connection.close()
    runtime_payload = [
        {
            "rollout_id": str(row["rollout_id"]),
            "instance_id": str(row["instance_id"]),
            "base_model_id": str(row["base_model_id"]),
            "base_model_sha256": str(row["base_model_sha256"]),
            "adapter_id": str(row["adapter_id"]),
            "adapter_sha256": str(row["adapter_sha256"]),
            "configuration_sha256": str(row["configuration_sha256"]),
            "status": str(row["status"]),
            "integrity_state": str(row["integrity_state"]),
        }
        for row in runtime_rows
    ]
    incomplete = 0
    active_canary = 0
    expired = 0
    unavailable = 0
    mismatch = 0
    disagreement = 0
    rollback_required = 0
    now = utc_now()
    for row in rows:
        state = str(row["state"])
        phase = str(row["transition_phase"]) if row["transition_phase"] is not None else None
        if state in TERMINAL_STATES and phase != "rollback_required":
            continue
        if phase is not None:
            incomplete += 1
        if phase == "rollback_required":
            rollback_required += 1
        if state == "canary_running":
            active_canary += 1
            expiry = row["canary_expires_at"]
            if expiry is not None:
                try:
                    expired += int(parse_utc_iso(str(expiry)) <= now)
                except ValueError:
                    expired += 1
        path = Path(str(row["candidate_artifact_path"]))
        if not path.is_file() or path.is_symlink():
            unavailable += 1
        elif _sha256_file(path) != str(row["candidate_sha256"]):
            mismatch += 1
        if (
            state == "active"
            and str(row["candidate_type"]) == "prompt_overlay"
            and active_prompt.get(str(row["target_id"])) != str(row["candidate_sha256"])
        ):
            disagreement += 1
        if (
            state == "canary_running"
            and str(row["candidate_type"]) == "prompt_overlay"
            and active_prompt.get(str(row["target_id"])) == str(row["candidate_sha256"])
        ):
            disagreement += 1
    runtime_rollback_required = sum(
        1 for item in runtime_payload if item["status"] == "rollback_required"
    )
    rollback_required += runtime_rollback_required
    unsafe = bool(
        incomplete or expired or unavailable or mismatch or disagreement or rollback_required
    )
    status = "degraded" if unsafe else ("disabled" if not enabled else "ok")
    return {
        **base,
        "status": status,
        "incomplete_transition_count": incomplete,
        "active_canary_count": active_canary,
        "expired_canary_count": expired,
        "candidate_unavailable_count": unavailable,
        "candidate_hash_mismatch_count": mismatch,
        "pointer_database_disagreement_count": disagreement,
        "rollback_required_count": rollback_required,
        "candidate_instances": runtime_payload,
        "candidate_integrity_state": (
            "mismatch"
            if any(item["integrity_state"] == "mismatch" for item in runtime_payload)
            else ("verified" if runtime_payload else "unknown")
        ),
        "rollout_state": (
            "rollback_required"
            if rollback_required
            else ("canary_running" if active_canary else "inactive")
        ),
        "rollback_required": bool(rollback_required),
        "action": (
            "run april evolve rollout list, then inspect and rollback unsafe rollouts"
            if unsafe
            else None
        ),
    }
