from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import yaml

from april_common.settings import AprilSettings
from services.evolution.write_guard import EvolutionWriteGuard
from services.memory.sqlite_memory import SqliteMemory
from skills.playbooks.miner import PlaybookMiner


@dataclass(slots=True)
class MiningReport:
    candidate_ids: list[str] = field(default_factory=list)
    candidate_paths: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "candidates": self.candidate_ids,
            "paths": self.candidate_paths,
            "details": self.details[:50],
        }


async def mine_playbook_candidates(
    memory: SqliteMemory,
    settings: AprilSettings,
    *,
    guard: EvolutionWriteGuard,
    max_conversations: int = 5,
) -> MiningReport:
    """D3: mine playbook candidates from successful local tool sequences.

    Conversations with two or more successfully executed tool calls become
    candidate playbooks written as data under data/playbooks (via the write
    fence). Candidates are never auto-adopted: status stays ``candidate`` and
    Level 3+ candidates additionally need the adoption approval flow before
    they can ever run from a trigger.
    """
    report = MiningReport()
    rows = await memory.database.fetchall(
        """
        SELECT conversation_id, tool, args_json, status
        FROM tool_calls
        WHERE conversation_id IS NOT NULL AND status = 'executed'
        ORDER BY conversation_id, created_at
        """
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
    for conversation_id in sorted(by_conversation)[:max_conversations]:
        calls = by_conversation[conversation_id]
        candidate = miner.mine(calls, name=f"Mined from conversation {conversation_id[:8]}")
        if candidate is None:
            continue
        if candidate.id in report.candidate_ids:
            continue
        target = settings.playbooks_path / f"{candidate.id}.yaml"
        written = guard.write_text(
            target,
            yaml.safe_dump(candidate.model_dump(), sort_keys=True),
        )
        report.candidate_ids.append(candidate.id)
        report.candidate_paths.append(str(written))
        report.details.append(
            f"mined {candidate.id} from conversation {conversation_id} "
            f"({len(candidate.steps)} steps, status=candidate)"
        )
    return report
