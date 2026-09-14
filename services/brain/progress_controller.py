"""Deterministic no-progress detection for bounded agent runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

ProgressAction = Literal["record", "warn", "stop"]


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    action_name: str
    argument_digest: str
    input_state_digest: str
    result_status: str
    evidence_digest: str
    new_evidence: bool = False

    @classmethod
    def from_values(
        cls,
        *,
        action_name: str,
        normalized_arguments: object,
        input_state_digest: str,
        result_status: str,
        evidence_digest: str,
        new_evidence: bool = False,
    ) -> ProgressEvent:
        arguments = json.dumps(
            normalized_arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return cls(
            action_name=action_name,
            argument_digest=hashlib.sha256(arguments.encode()).hexdigest(),
            input_state_digest=input_state_digest,
            result_status=result_status,
            evidence_digest=evidence_digest,
            new_evidence=new_evidence,
        )

    @property
    def fingerprint(self) -> str:
        payload = {
            "action": self.action_name,
            "arguments": self.argument_digest,
            "state": self.input_state_digest,
            "status": self.result_status,
            "evidence": self.evidence_digest,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class ProgressDecision:
    action: ProgressAction
    duplicate_count: int
    reason: str


@dataclass(slots=True)
class ProgressController:
    duplicate_warning_threshold: int = 2
    max_no_progress_cycles: int = 3
    _counts: dict[str, int] = field(default_factory=dict)

    def observe(self, event: ProgressEvent) -> ProgressDecision:
        fingerprint = event.fingerprint
        if event.new_evidence:
            return ProgressDecision("record", 0, "new_evidence")
        count = self._counts.get(fingerprint, 0) + 1
        self._counts[fingerprint] = count
        if count >= self.max_no_progress_cycles:
            return ProgressDecision("stop", count, "no_progress_limit")
        if count >= self.duplicate_warning_threshold:
            return ProgressDecision("warn", count, "duplicate_action_unchanged_state")
        return ProgressDecision("record", count, "first_observation")
