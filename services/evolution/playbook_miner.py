from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import yaml

from april_common.settings import AprilSettings
from april_common.time import utc_now
from services.evolution.write_guard import EvolutionWriteGuard
from services.memory.sqlite_memory import SqliteMemory
from skills.playbooks.loader import PlaybookLoader
from skills.playbooks.miner import PlaybookMiner
from skills.playbooks.schema import PlaybookDefinition
from skills.registry import default_registry


@dataclass(slots=True)
class MiningReport:
    candidate_ids: list[str] = field(default_factory=list)
    candidate_paths: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)
    support: dict[str, int] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "candidates": self.candidate_ids,
            "paths": self.candidate_paths,
            "details": self.details[:50],
            "support": self.support,
        }


async def mine_playbook_candidates(
    memory: SqliteMemory,
    settings: AprilSettings,
    *,
    guard: EvolutionWriteGuard,
    support_threshold: int = 3,
    lookback_days: int = 14,
    max_conversations: int = 500,
) -> MiningReport:
    """D3: mine playbook candidates from successful local tool sequences.

    Successful tool-call sequences must recur at least ``support_threshold``
    times inside the bounded lookback window. Candidates are never auto-adopted:
    status stays ``candidate`` and Level 3+ candidates additionally need the
    adoption approval flow before they can ever run from a trigger.
    """
    report = MiningReport()
    since = (utc_now() - timedelta(days=max(1, lookback_days))).isoformat().replace(
        "+00:00", "Z"
    )
    rows = await memory.database.fetchall(
        """
        SELECT conversation_id, tool, args_json, status
        FROM tool_calls
        WHERE conversation_id IS NOT NULL
          AND status = 'executed'
          AND created_at >= ?
        ORDER BY conversation_id, created_at
        """,
        (since,),
    )
    by_conversation: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        try:
            args = json.loads(row["args_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(args, dict):
            continue
        by_conversation.setdefault(str(row["conversation_id"]), []).append(
            {"tool": str(row["tool"]), "args": args, "status": str(row["status"])}
        )
    miner = PlaybookMiner()
    sequences = [
        by_conversation[conversation_id]
        for conversation_id in sorted(by_conversation)[:max_conversations]
    ]
    existing_ids = await _existing_playbook_ids(memory, settings)
    known_tools = {definition.name for definition in default_registry().list()}
    candidates = miner.mine_frequent(
        sequences,
        support_threshold=support_threshold,
        existing_ids=existing_ids,
        known_tools=known_tools,
    )
    for candidate in candidates:
        target = settings.playbooks_path / f"{candidate.id}.yaml"
        written = guard.write_text(
            target,
            yaml.safe_dump(candidate.model_dump(), sort_keys=True),
        )
        report.candidate_ids.append(candidate.id)
        report.candidate_paths.append(str(written))
        support = _candidate_support(candidate)
        report.support[candidate.id] = support
        report.details.append(
            f"mined {candidate.id} with support {support} "
            f"({len(candidate.steps)} steps, status=candidate)"
        )
    return report


async def _existing_playbook_ids(
    memory: SqliteMemory, settings: AprilSettings
) -> set[str]:
    ids = {playbook.id for playbook in PlaybookLoader(settings.playbooks_path).list()}
    rows = await memory.database.fetchall("SELECT id FROM playbooks")
    ids.update(str(row["id"]) for row in rows)
    return ids


def _candidate_support(candidate: PlaybookDefinition) -> int:
    if not candidate.steps:
        return 0
    reason = candidate.steps[0].reason
    marker = "support="
    if marker not in reason:
        return 0
    tail = reason.split(marker, 1)[1]
    digits = []
    for char in tail:
        if not char.isdigit():
            break
        digits.append(char)
    return int("".join(digits)) if digits else 0
