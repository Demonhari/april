"""Small orchestration controller for bounded progress and completion decisions."""

from __future__ import annotations

from dataclasses import dataclass

from services.brain.coding_workflow import CodingCompletionGate, CompletionDecision
from services.brain.progress_controller import ProgressController, ProgressDecision, ProgressEvent
from services.brain.repository_state import RepositoryState, VerificationEvidence
from services.brain.task_contract import TaskContract


@dataclass(slots=True)
class RunController:
    """Keep deterministic progress and verification policy outside the model."""

    contract: TaskContract
    progress: ProgressController
    completion: CodingCompletionGate
    replan_attempts: int = 0

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
