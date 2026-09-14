"""Small orchestration controller for bounded progress and completion decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from services.brain.coding_workflow import CodingCompletionGate, CompletionDecision
from services.brain.progress_controller import ProgressController, ProgressDecision, ProgressEvent
from services.brain.repository_state import RepositoryState, VerificationEvidence
from services.brain.task_contract import TaskContract

_CODING_STAGES = {
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
}


@dataclass(slots=True)
class RunController:
    """Keep deterministic progress and verification policy outside the model."""

    contract: TaskContract
    progress: ProgressController
    completion: CodingCompletionGate
    replan_attempts: int = 0
    latest_evidence: VerificationEvidence | None = None
    last_evidence_digest: str | None = None

    @classmethod
    def for_contract(cls, contract: TaskContract) -> RunController:
        return cls(
            contract=contract,
            progress=ProgressController(),
            completion=CodingCompletionGate(),
        )

    def observe(self, event: ProgressEvent) -> ProgressDecision:
        return self.progress.observe(event)

    def request_replan(self) -> bool:
        if self.replan_attempts >= self.contract.maximum_replan_attempts:
            return False
        self.replan_attempts += 1
        return True

    def completion_decision(
        self,
        *,
        state: RepositoryState,
        evidence: VerificationEvidence | None,
    ) -> CompletionDecision:
        return self.completion.can_complete(self.contract, state=state, evidence=evidence)

    @staticmethod
    def trusted_feedback(decision: ProgressDecision) -> str:
        if decision.action == "warn":
            return (
                "The previous action repeated against the same state and evidence. "
                "Gather new evidence or change strategy before repeating it."
            )
        return "APRIL stopped this run after bounded no-progress repetition."

    def snapshot(self) -> dict[str, Any]:
        """Return non-secret durable state needed to resume this run."""

        return {
            "contract": self.contract.model_dump(mode="json"),
            "replan_attempts": self.replan_attempts,
            "progress_counts": dict(self.progress._counts),
            "last_verified_state_digest": self.completion.last_verified_state_digest,
            "completion_stage": self.completion.stage,
            "latest_evidence": (
                asdict(self.latest_evidence) if self.latest_evidence is not None else None
            ),
            "last_evidence_digest": self.last_evidence_digest,
        }

    @classmethod
    def restore(cls, snapshot: dict[str, Any]) -> RunController:
        contract = TaskContract.model_validate(snapshot.get("contract", {}))
        progress = ProgressController()
        counts = snapshot.get("progress_counts")
        if isinstance(counts, dict):
            progress._counts = {
                str(key): int(value) for key, value in counts.items() if isinstance(value, int)
            }
        stage = snapshot.get("completion_stage", "preflight")
        if stage not in _CODING_STAGES:
            raise ValueError("invalid persisted coding completion stage")
        completion = CodingCompletionGate(
            stage=stage,
            last_verified_state_digest=(
                str(snapshot["last_verified_state_digest"])
                if snapshot.get("last_verified_state_digest")
                else None
            ),
        )
        raw_evidence = snapshot.get("latest_evidence")
        evidence = VerificationEvidence(**raw_evidence) if isinstance(raw_evidence, dict) else None
        return cls(
            contract=contract,
            progress=progress,
            completion=completion,
            replan_attempts=max(0, int(snapshot.get("replan_attempts", 0))),
            latest_evidence=evidence,
            last_evidence_digest=(
                str(snapshot["last_evidence_digest"])
                if snapshot.get("last_evidence_digest")
                else None
            ),
        )
