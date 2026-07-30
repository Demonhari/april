"""Safe database and hash-chained audit events for rollout transitions."""

from __future__ import annotations

from typing import Any

from april_common.time import utc_now_iso
from services.evolution.rollout_base import RolloutServiceBase
from services.evolution.rollout_models import RolloutRecord
from services.evolution.rollout_policy import (
    _canonical_json,
    _reason_code,
)


class RolloutAudit(RolloutServiceBase):
    async def _event_tx(
        self,
        connection: Any,
        rollout_id: str,
        event_type: str,
        *,
        reason_code: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        self.guard.validate_table("evolution_rollout_events")
        await connection.execute(
            """
            INSERT INTO evolution_rollout_events(
                rollout_id, event_type, reason_code, safe_summary_json, created_at
            )
            VALUES(?, ?, ?, ?, ?)
            """,
            (
                rollout_id,
                event_type[:96],
                _reason_code(reason_code) if reason_code else None,
                _canonical_json(summary or {}),
                utc_now_iso(),
            ),
        )

    def _audit(
        self,
        event_type: str,
        record: RolloutRecord,
        *,
        reason: str | None = None,
        automatic: bool = False,
    ) -> None:
        if self.audit is None:
            return
        self.audit.write(
            {
                "event_type": event_type,
                "actor": "april-core" if automatic else "local-user",
                "rollout_id": record.id,
                "candidate_type": record.candidate_type,
                "candidate_id": record.candidate_id,
                "candidate_sha256": record.candidate_sha256,
                "baseline_id": record.baseline_id,
                "baseline_sha256": record.baseline_sha256,
                "state": record.state,
                "reason_code": _reason_code(reason) if reason else None,
                "automatic": automatic,
            }
        )
