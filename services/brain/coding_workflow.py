"""Bounded coding stages and a state-bound completion gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from services.brain.repository_state import RepositoryState, VerificationEvidence
from services.brain.task_contract import TaskContract

CodingStage = Literal[
    "preflight",
    "investigate",
    "hypothesis",
    "plan",
    "propose_patch",
    "approve_patch",
    "apply",
    "verify",
    "review",
    "complete",
]


@dataclass(frozen=True, slots=True)
class CompletionDecision:
    allowed: bool
    reason: str
    stage: CodingStage


@dataclass(slots=True)
class CodingCompletionGate:
    stage: CodingStage = "preflight"
    last_verified_state_digest: str | None = None

    def mark_stage(self, stage: CodingStage) -> None:
        self.stage = stage

    def mark_verification(self, evidence: VerificationEvidence) -> None:
        self.last_verified_state_digest = evidence.repository_state_digest
        self.stage = "review"

    def can_complete(
        self,
        contract: TaskContract,
        *,
        state: RepositoryState,
        evidence: VerificationEvidence | None,
    ) -> CompletionDecision:
        if contract.can_finish_without_current_verification():
            return CompletionDecision(True, "verification_not_required", "complete")
        if evidence is None:
            return CompletionDecision(False, "current_verification_required", "verify")
        if not evidence.is_current(state):
            return CompletionDecision(False, "verification_evidence_is_stale", "verify")
        if evidence.result_status != "pass" or evidence.exit_status != 0:
            return CompletionDecision(False, "verification_failed", "verify")
        self.last_verified_state_digest = state.digest
        self.stage = "complete"
        return CompletionDecision(True, "verified_current_repository_state", "complete")
