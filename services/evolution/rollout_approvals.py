"""Exact Level 4 approval creation and validation for rollout gates."""

from __future__ import annotations

import json
import uuid
from typing import Any, Literal

from april_common.time import parse_utc_iso, utc_now
from services.evolution.rollout_base import RolloutServiceBase
from services.evolution.rollout_models import RolloutBlocked, RolloutRecord
from services.permissions.approvals import ApprovalStore, canonical_hash
from services.permissions.schemas import ApprovalRequest


class RolloutApprovals(RolloutServiceBase):
    async def request_approval(
        self,
        rollout_id: str,
        *,
        stage: Literal["canary", "activation"],
        approvals: ApprovalStore,
        actor: str = "local-user",
        request_id: str | None = None,
    ) -> str:
        """Explicit owner action that creates, but never approves, an exact L4 gate."""

        record = await self.require(rollout_id)
        expected_state = "shadow_passed" if stage == "canary" else "canary_passed"
        if record.state != expected_state:
            raise RolloutBlocked(f"{stage}_approval_not_available_from_{record.state}")
        tool, args = self._approval_action(record, stage)
        response = await approvals.create(
            ApprovalRequest(
                tool=tool,
                args=args,
                agent="local-operator",
                permission_level=4,
                risk_level="system_action",
                affected_paths=[record.candidate_id],
                expected_side_effects=[
                    (
                        "route a bounded fraction of eligible low-risk prompt requests"
                        if stage == "canary"
                        else "publish the reviewed prompt overlay as active"
                    )
                ],
                metadata={
                    "rollout_id": record.id,
                    "stage": stage,
                    "candidate_sha256": record.candidate_sha256,
                },
            ),
            actor=actor,
            request_id=request_id or str(uuid.uuid4()),
        )
        return response.approval_id

    async def _validate_approval_tx(
        self,
        connection: Any,
        *,
        approval_id: str,
        tool: str,
        args: dict[str, Any],
    ) -> None:
        cursor = await connection.execute(
            "SELECT * FROM approvals WHERE id = ?",
            (approval_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise RolloutBlocked("approval_not_found")
        try:
            stored_args = json.loads(str(row["args_json"]))
            metadata = json.loads(str(row["metadata_json"] or "{}"))
        except json.JSONDecodeError as exc:
            raise RolloutBlocked("approval_record_invalid") from exc
        if not isinstance(stored_args, dict) or not isinstance(metadata, dict):
            raise RolloutBlocked("approval_record_invalid")
        if (
            str(row["tool"]) != tool
            or stored_args != args
            or str(row["canonical_hash"]) != canonical_hash(tool, args, metadata)
            or int(row["permission_level"]) != 4
            or str(row["risk_level"]) != "system_action"
        ):
            raise RolloutBlocked("approval_action_mismatch")
        if str(row["status"]) != "approved":
            raise RolloutBlocked("approval_not_approved")
        try:
            if parse_utc_iso(str(row["expires_at"])) < utc_now():
                raise RolloutBlocked("approval_expired")
        except ValueError as exc:
            raise RolloutBlocked("approval_record_invalid") from exc

    def _approval_action(
        self,
        record: RolloutRecord,
        stage: Literal["canary", "activation"],
    ) -> tuple[str, dict[str, Any]]:
        return (
            f"evolution_rollout_{stage}",
            {
                "rollout_id": record.id,
                "candidate_id": record.candidate_id,
                "candidate_sha256": record.candidate_sha256,
                "baseline_id": record.baseline_id,
                "baseline_sha256": record.baseline_sha256,
                "configuration_sha256": record.configuration_sha256,
                "shadow_evidence_sha256": record.shadow_evidence_sha256,
            },
        )
