from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from services.evolution.write_guard import EvolutionWriteGuard
from services.memory.sqlite_memory import SqliteMemory


@dataclass(slots=True)
class ConsolidationReport:
    duplicates_merged: int = 0
    contradictions_resolved: int = 0
    details: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "duplicates_merged": self.duplicates_merged,
            "contradictions_resolved": self.contradictions_resolved,
            "details": self.details[:50],
        }


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


async def consolidate_memories(
    memory: SqliteMemory, *, guard: EvolutionWriteGuard
) -> ConsolidationReport:
    """D2: merge duplicate memories and adjudicate flagged contradictions.

    Nothing is deleted: duplicate losers and contradiction losers get
    ``superseded_by`` set, keepers get an auditable usage/confidence refresh,
    and contradiction pairs are marked resolved with the rule that decided
    them. All touched tables are validated against the evolution write fence.
    """
    guard.validate_table("memories")
    guard.validate_table("memory_contradictions")
    report = ConsolidationReport()

    # Duplicate merge: same kind + normalized content + project keeps the
    # earliest row; later rows are superseded by it.
    groups: dict[tuple[str, str, str | None], list[Any]] = {}
    for record in await memory.list_memories():
        groups.setdefault((record.kind, _normalized(record.content), record.project_id), []).append(
            record
        )
    for records in groups.values():
        if len(records) < 2:
            continue
        ordered = sorted(records, key=lambda record: record.created_at)
        keeper = ordered[0]
        best_confidence = max(record.confidence for record in ordered)
        for duplicate in ordered[1:]:
            if await memory.supersede_memory(duplicate.id, superseded_by=keeper.id):
                report.duplicates_merged += 1
                report.details.append(f"merged duplicate {duplicate.id} into {keeper.id}")
        await memory.refresh_memory(keeper.id, confidence=best_confidence)

    # Contradiction adjudication: higher confidence wins; ties go to the newer
    # statement. The loser is superseded, both rows stay.
    for pair in await memory.list_memory_contradictions(status="pending"):
        first = await memory.get_memory(pair.memory_id_a, include_inactive=True)
        second = await memory.get_memory(pair.memory_id_b, include_inactive=True)
        if first is None or second is None:
            await memory.resolve_memory_contradiction(
                pair.id, resolution="invalid: memory row missing"
            )
            continue
        if first.superseded_by is not None or second.superseded_by is not None:
            await memory.resolve_memory_contradiction(
                pair.id, resolution="already superseded elsewhere"
            )
            continue
        if first.confidence != second.confidence:
            winner, loser = (
                (first, second) if first.confidence > second.confidence else (second, first)
            )
            rule = "higher confidence"
        else:
            winner, loser = (
                (first, second) if first.created_at >= second.created_at else (second, first)
            )
            rule = "newer statement on equal confidence"
        await memory.supersede_memory(loser.id, superseded_by=winner.id)
        await memory.resolve_memory_contradiction(
            pair.id, resolution=f"winner={winner.id} rule={rule}"
        )
        report.contradictions_resolved += 1
        report.details.append(
            f"contradiction {pair.id}: kept {winner.id} ({rule}), superseded {loser.id}"
        )
    return report
