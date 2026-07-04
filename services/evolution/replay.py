from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from services.memory.sqlite_memory import SqliteMemory


@dataclass(frozen=True, slots=True)
class ReplayItem:
    kind: str  # negative_feedback | correction | approval_denial | normal_sample
    ref_id: str
    summary: str
    created_at: str


@dataclass(slots=True)
class ReplayReport:
    items: list[ReplayItem] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.kind] = counts.get(item.kind, 0) + 1
        return counts

    def to_payload(self) -> dict[str, Any]:
        return {
            "counts": self.counts,
            "items": [
                {
                    "kind": item.kind,
                    "ref_id": item.ref_id,
                    "summary": item.summary[:200],
                    "created_at": item.created_at,
                }
                for item in self.items
            ],
        }


async def collect_replay_samples(
    memory: SqliteMemory,
    *,
    seed: int,
    per_kind_limit: int = 10,
    normal_sample_size: int = 3,
) -> ReplayReport:
    """D1: gather the runs worth re-examining tonight.

    Negative feedback, corrections, and approval denials are collected in full
    (bounded), plus a small deterministic random sample of normal runs so the
    Dreamer also sees what went right. The RNG is seeded from the run date, so
    the same night over the same data replays identically.
    """
    report = ReplayReport()
    for event in await memory.list_feedback_events(limit=per_kind_limit * 2):
        if event.rating != "bad" or len_of_kind(report, "negative_feedback") >= per_kind_limit:
            continue
        report.items.append(
            ReplayItem(
                kind="negative_feedback",
                ref_id=event.id,
                summary=event.reason or "negative feedback without reason",
                created_at=event.created_at,
            )
        )
    for record in await memory.list_memories():
        if record.kind != "correction":
            continue
        if len_of_kind(report, "correction") >= per_kind_limit:
            break
        report.items.append(
            ReplayItem(
                kind="correction",
                ref_id=record.id,
                summary=record.content,
                created_at=record.created_at,
            )
        )
    denials = await memory.database.fetchall(
        "SELECT id, tool, created_at FROM approvals WHERE status = 'denied' "
        "ORDER BY created_at DESC LIMIT ?",
        (per_kind_limit,),
    )
    for row in denials:
        report.items.append(
            ReplayItem(
                kind="approval_denial",
                ref_id=str(row["id"]),
                summary=f"approval denied for tool {row['tool']}",
                created_at=str(row["created_at"]),
            )
        )
    normal_rows = await memory.database.fetchall(
        "SELECT id, agent, summary, created_at FROM agent_runs "
        "WHERE status = 'ok' ORDER BY created_at DESC LIMIT 50"
    )
    rng = random.Random(seed)
    sample = rng.sample(list(normal_rows), min(normal_sample_size, len(normal_rows)))
    for row in sorted(sample, key=lambda item: str(item["created_at"])):
        report.items.append(
            ReplayItem(
                kind="normal_sample",
                ref_id=str(row["id"]),
                summary=f"{row['agent']}: {row['summary'] or 'ok run'}",
                created_at=str(row["created_at"]),
            )
        )
    return report


def len_of_kind(report: ReplayReport, kind: str) -> int:
    return sum(1 for item in report.items if item.kind == kind)
