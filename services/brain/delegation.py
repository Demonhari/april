"""Bounded internal specialist contracts; no authority is added by delegation."""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

SpecialistRole = Literal["investigator", "implementer", "reviewer", "researcher"]


class DelegationCeiling(BaseModel):
    allowed_tools: frozenset[str] = frozenset()
    maximum_risk_level: int = Field(default=1, ge=0, le=4)
    project_id: str | None = None
    project_root: str | None = None
    maximum_depth: int = Field(default=1, ge=0, le=2)

    def intersect(self, specialist: DelegationCeiling) -> DelegationCeiling:
        if (
            self.project_root
            and specialist.project_root
            and self.project_root != specialist.project_root
        ):
            raise ValueError("specialist project root exceeds the parent scope")
        else:
            project_root = specialist.project_root or self.project_root
        if self.project_id and specialist.project_id and self.project_id != specialist.project_id:
            raise ValueError("specialist project exceeds the parent scope")
        else:
            project_id = specialist.project_id or self.project_id
        return DelegationCeiling(
            allowed_tools=self.allowed_tools & specialist.allowed_tools,
            maximum_risk_level=min(self.maximum_risk_level, specialist.maximum_risk_level),
            project_id=project_id,
            project_root=project_root,
            maximum_depth=min(self.maximum_depth, specialist.maximum_depth),
        )


class SpecialistTask(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_run_id: str
    role: SpecialistRole
    goal: str = Field(min_length=1, max_length=8_192)
    evidence_references: tuple[str, ...] = ()
    project_id: str | None = None
    capability_ceiling: DelegationCeiling
    max_iterations: int = Field(default=3, ge=1, le=10)
    status: Literal["queued", "running", "completed", "incomplete", "failed"] = "queued"
    deliverable_schema: str = "structured_evidence"

    @classmethod
    def bounded(
        cls,
        *,
        parent_run_id: str,
        role: SpecialistRole,
        goal: str,
        parent: DelegationCeiling,
        specialist: DelegationCeiling,
        global_policy: DelegationCeiling | None = None,
    ) -> SpecialistTask:
        ceiling = parent.intersect(specialist)
        if global_policy is not None:
            ceiling = ceiling.intersect(global_policy)
        if parent.maximum_depth <= 0:
            raise ValueError("maximum delegation depth exceeded")
        ceiling.maximum_depth = max(0, ceiling.maximum_depth - 1)
        return cls(
            parent_run_id=parent_run_id,
            role=role,
            goal=goal,
            project_id=ceiling.project_id,
            capability_ceiling=ceiling,
        )
