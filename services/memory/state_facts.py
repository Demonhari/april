"""Explicit current-state facts with exact-key supersession semantics."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal, cast
from uuid import uuid4

from pydantic import BaseModel, Field

from services.memory.database import Database

FactStatus = Literal["current", "historical", "retracted"]
FactOrigin = Literal["user", "reflection", "import", "system_observation"]


class StateFactKey(BaseModel):
    owner_scope: str
    project_id: str | None = None
    entity: str
    attribute: str
    cardinality: Literal["single", "many"] = "single"


class StateFact(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    key: StateFactKey
    value: object
    evidence_reference: str
    confidence: float = Field(default=0.7, ge=0, le=1)
    observed_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: FactStatus = "current"
    superseded_by: str | None = None
    origin: FactOrigin = "system_observation"


class StateFactStore:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def put(self, fact: StateFact) -> StateFact:
        async with self.database.transaction() as connection:
            current = None
            superseded_id: str | None = None
            if fact.key.cardinality == "single":
                cursor = await connection.execute(
                    """
                    SELECT * FROM state_facts
                    WHERE owner_scope = ? AND project_id IS ? AND entity = ?
                      AND attribute = ? AND cardinality = 'single' AND status = 'current'
                    ORDER BY observed_at DESC LIMIT 1
                    """,
                    (
                        fact.key.owner_scope,
                        fact.key.project_id,
                        fact.key.entity,
                        fact.key.attribute,
                    ),
                )
                current = await cursor.fetchone()
                if current is not None and str(current["observed_at"]) > fact.observed_at:
                    fact = fact.model_copy(update={"status": "historical"})
                elif current is not None:
                    superseded_id = str(current["id"])
                else:
                    superseded_id = None
            else:
                superseded_id = None
            await connection.execute(
                """
                INSERT INTO state_facts(
                    id, owner_scope, project_id, entity, attribute, cardinality,
                    value_json, evidence_reference, confidence, observed_at,
                    status, superseded_by, origin
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact.id,
                    fact.key.owner_scope,
                    fact.key.project_id,
                    fact.key.entity,
                    fact.key.attribute,
                    fact.key.cardinality,
                    json.dumps(fact.value, sort_keys=True, ensure_ascii=False),
                    fact.evidence_reference,
                    fact.confidence,
                    fact.observed_at,
                    fact.status,
                    fact.superseded_by,
                    fact.origin,
                ),
            )
            if current is not None and fact.status == "current":
                await connection.execute(
                    "UPDATE state_facts SET status = 'historical', superseded_by = ? WHERE id = ?",
                    (fact.id, superseded_id),
                )
        return fact

    async def current(self, key: StateFactKey) -> list[StateFact]:
        rows = await self.database.fetchall(
            """
            SELECT * FROM state_facts
            WHERE owner_scope = ? AND project_id IS ? AND entity = ?
              AND attribute = ? AND cardinality = ? AND status = 'current'
            ORDER BY observed_at DESC
            """,
            (key.owner_scope, key.project_id, key.entity, key.attribute, key.cardinality),
        )
        return [_row_to_fact(row) for row in rows]


def _row_to_fact(row: object) -> StateFact:
    mapping = dict(cast(Mapping[str, object], row))
    return StateFact(
        id=str(mapping["id"]),
        key=StateFactKey(
            owner_scope=str(mapping["owner_scope"]),
            project_id=mapping["project_id"],
            entity=str(mapping["entity"]),
            attribute=str(mapping["attribute"]),
            cardinality=str(mapping["cardinality"]),
        ),
        value=json.loads(str(mapping["value_json"])),
        evidence_reference=str(mapping["evidence_reference"]),
        confidence=float(cast(Any, mapping["confidence"])),
        observed_at=str(mapping["observed_at"]),
        status=str(mapping["status"]),
        superseded_by=mapping["superseded_by"],
        origin=str(mapping["origin"]),
    )
