"""Reviewed experience lifecycle; candidates are never trusted prompt policy."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal, cast
from uuid import uuid4

from pydantic import BaseModel, Field

from services.memory.database import Database

LessonStatus = Literal["candidate", "approved", "rejected", "superseded"]


class ExperienceLesson(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    project_id: str | None = None
    task_signature: str
    lesson: str = Field(min_length=1, max_length=8_192)
    supporting_evidence_ids: tuple[str, ...] = ()
    confidence: float = Field(default=0.5, ge=0, le=1)
    status: LessonStatus = "candidate"
    creation_reason: str


class LessonStore:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def create_candidate(self, lesson: ExperienceLesson) -> ExperienceLesson:
        if lesson.status != "candidate":
            raise ValueError("new experience lessons must start as candidates")
        await self.database.execute(
            """
            INSERT INTO experience_lessons(
                id, project_id, task_signature, lesson, supporting_evidence_ids_json,
                confidence, status, creation_reason, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'candidate', ?, datetime('now'), datetime('now'))
            """,
            (
                lesson.id,
                lesson.project_id,
                lesson.task_signature,
                lesson.lesson,
                json.dumps(list(lesson.supporting_evidence_ids), sort_keys=True),
                lesson.confidence,
                lesson.creation_reason,
            ),
        )
        return lesson

    async def transition(self, lesson_id: str, status: LessonStatus) -> None:
        if status == "candidate":
            raise ValueError("a lesson cannot transition back to candidate")
        await self.database.execute(
            "UPDATE experience_lessons SET status = ?, updated_at = datetime('now') WHERE id = ?",
            (status, lesson_id),
        )

    async def approved(self, *, project_id: str | None = None) -> list[ExperienceLesson]:
        rows = await self.database.fetchall(
            "SELECT * FROM experience_lessons WHERE status = 'approved' AND project_id IS ?",
            (project_id,),
        )
        return [_row_to_lesson(row) for row in rows]

    async def list(
        self, *, status: LessonStatus | None = None, project_id: str | None = None
    ) -> list[ExperienceLesson]:
        if status is None:
            rows = await self.database.fetchall(
                "SELECT * FROM experience_lessons WHERE project_id IS ? ORDER BY updated_at DESC",
                (project_id,),
            )
        else:
            rows = await self.database.fetchall(
                "SELECT * FROM experience_lessons WHERE status = ? AND project_id IS ? "
                "ORDER BY updated_at DESC",
                (status, project_id),
            )
        return [_row_to_lesson(row) for row in rows]

    async def get(self, lesson_id: str) -> ExperienceLesson | None:
        row = await self.database.fetchone(
            "SELECT * FROM experience_lessons WHERE id = ?", (lesson_id,)
        )
        return _row_to_lesson(row) if row is not None else None


def _row_to_lesson(row: object) -> ExperienceLesson:
    mapping = dict(cast(Mapping[str, object], row))
    return ExperienceLesson(
        id=str(mapping["id"]),
        project_id=mapping["project_id"],
        task_signature=str(mapping["task_signature"]),
        lesson=str(mapping["lesson"]),
        supporting_evidence_ids=tuple(json.loads(str(mapping["supporting_evidence_ids_json"]))),
        confidence=float(cast(Any, mapping["confidence"])),
        status=str(mapping["status"]),
        creation_reason=str(mapping["creation_reason"]),
    )
