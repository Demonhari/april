from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field

from april_common.time import utc_now_iso
from services.brain.schemas import BrainDecision

TaskStatus = Literal["planned", "running", "completed", "pending_approval", "error"]
TaskStepStatus = Literal["pending", "running", "completed", "skipped", "error"]


class TaskStep(BaseModel):
    index: int = Field(ge=1)
    title: str = Field(min_length=1)
    status: TaskStepStatus = "pending"
    tool_hint: str | None = None


class TaskPlan(BaseModel):
    id: str
    conversation_id: str
    request_id: str
    intent: str
    agent: str
    model_id: str
    steps: list[TaskStep]
    status: TaskStatus = "planned"
    created_at: str


def task_plan_from_decision(
    decision: BrainDecision,
    *,
    conversation_id: str,
    request_id: str,
) -> TaskPlan:
    raw_steps = decision.task_steps or [decision.decision_summary]
    steps = [
        TaskStep(
            index=index,
            title=title.strip()[:200],
            tool_hint=decision.tools_needed[index - 1]
            if index - 1 < len(decision.tools_needed)
            else None,
        )
        for index, title in enumerate(raw_steps[:8], start=1)
        if title.strip()
    ]
    if not steps:
        steps = [TaskStep(index=1, title="Answer request")]
    return TaskPlan(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        request_id=request_id,
        intent=decision.intent,
        agent=decision.agent,
        model_id=decision.model_id,
        steps=steps,
        status="planned",
        created_at=utc_now_iso(),
    )
