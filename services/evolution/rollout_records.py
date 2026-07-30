from __future__ import annotations

import json
from typing import Any

from services.evolution.rollout_models import RolloutBlocked, RolloutRecord


def _record_from_row(row: Any) -> RolloutRecord:
    try:
        metrics = json.loads(str(row["metrics_json"] or "{}"))
        previous = (
            json.loads(str(row["previous_active_artifact_json"]))
            if row["previous_active_artifact_json"] is not None
            else None
        )
    except json.JSONDecodeError as exc:
        raise RolloutBlocked("rollout_record_invalid") from exc
    if not isinstance(metrics, dict) or (previous is not None and not isinstance(previous, dict)):
        raise RolloutBlocked("rollout_record_invalid")
    return RolloutRecord(
        id=str(row["id"]),
        candidate_type=str(row["candidate_type"]),  # type: ignore[arg-type]
        target_id=str(row["target_id"]),
        candidate_id=str(row["candidate_id"]),
        candidate_sha256=str(row["candidate_sha256"]),
        candidate_artifact_path=str(row["candidate_artifact_path"]),
        baseline_id=str(row["baseline_id"]),
        baseline_sha256=str(row["baseline_sha256"]),
        baseline_artifact_path=(
            str(row["baseline_artifact_path"])
            if row["baseline_artifact_path"] is not None
            else None
        ),
        state=str(row["state"]),  # type: ignore[arg-type]
        configuration_sha256=str(row["configuration_sha256"]),
        shadow_dataset_sha256=(
            str(row["shadow_dataset_sha256"]) if row["shadow_dataset_sha256"] is not None else None
        ),
        shadow_evidence_sha256=(
            str(row["shadow_evidence_sha256"])
            if row["shadow_evidence_sha256"] is not None
            else None
        ),
        requested_minimum_samples=int(row["requested_minimum_samples"]),
        completed_sample_count=int(row["completed_sample_count"]),
        canary_traffic_fraction=float(row["canary_traffic_fraction"]),
        canary_max_eligible_turns=(
            int(row["canary_max_eligible_turns"])
            if row["canary_max_eligible_turns"] is not None
            else None
        ),
        canary_eligible_turn_count=int(row["canary_eligible_turn_count"]),
        canary_selected_turn_count=int(row["canary_selected_turn_count"]),
        canary_expires_at=(
            str(row["canary_expires_at"]) if row["canary_expires_at"] is not None else None
        ),
        metrics=metrics,
        reason_code=str(row["reason_code"]) if row["reason_code"] is not None else None,
        canary_approval_id=(
            str(row["canary_approval_id"]) if row["canary_approval_id"] is not None else None
        ),
        activation_approval_id=(
            str(row["activation_approval_id"])
            if row["activation_approval_id"] is not None
            else None
        ),
        previous_active_artifact=previous,
        transition_phase=(
            str(row["transition_phase"]) if row["transition_phase"] is not None else None
        ),
        shadow_job_id=(str(row["shadow_job_id"]) if row["shadow_job_id"] is not None else None),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        started_at=str(row["started_at"]) if row["started_at"] is not None else None,
        completed_at=(str(row["completed_at"]) if row["completed_at"] is not None else None),
        rolled_back_at=(str(row["rolled_back_at"]) if row["rolled_back_at"] is not None else None),
        version=int(row["version"]),
    )
