from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import yaml

from april_common.settings import AprilSettings
from april_common.time import utc_now
from services.evolution.write_guard import EvolutionWriteGuard
from services.memory.policy import MemoryPolicy
from services.memory.sqlite_memory import SqliteMemory
from skills.playbooks.loader import PlaybookLoader
from skills.playbooks.miner import PlaybookMiner
from skills.playbooks.schema import PlaybookDefinition
from skills.registry import default_registry

# Trigger suggestions come only from short, non-sensitive user messages.
_MAX_TRIGGER_CHARS = 120
_MAX_TRIGGER_EXAMPLES = 3
# Learned playbooks whose steps stay below this level may be auto-adopted;
# everything at or above it requires the explicit adoption approval flow.
_ADOPTION_APPROVAL_LEVEL = 3


@dataclass(slots=True)
class MiningReport:
    candidate_ids: list[str] = field(default_factory=list)
    candidate_paths: list[str] = field(default_factory=list)
    adopted_ids: list[str] = field(default_factory=list)
    approval_required_ids: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)
    support: dict[str, int] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "candidates": self.candidate_ids,
            "paths": self.candidate_paths,
            "adopted": self.adopted_ids,
            "approval_required": self.approval_required_ids,
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
    auto_adopt: bool = True,
) -> MiningReport:
    """D3: mine playbook candidates from successful local tool sequences.

    Frequent *contiguous subsequences* of successful tool calls must recur in at
    least ``support_threshold`` conversations inside the bounded lookback
    window. Trigger examples are suggested only from short, non-sensitive user
    messages of the supporting conversations. Adoption policy:

    * L0-L2 candidates with a safe, unambiguous trigger may be auto-adopted
      (``auto_adopt=True``); each is audited via the fenced write path.
    * Candidates without a safe trigger stay ``candidate`` until the user
      provides one.
    * Level 3+ candidates always stay ``candidate`` and require the explicit
      adoption approval flow; per-run L3+ steps still raise their own
      exact-action approvals when the playbook executes.
    """
    report = MiningReport()
    since = (utc_now() - timedelta(days=max(1, lookback_days))).isoformat().replace("+00:00", "Z")
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
    conversation_ids = sorted(by_conversation)[:max_conversations]
    sequences = [by_conversation[conversation_id] for conversation_id in conversation_ids]
    existing_ids = await _existing_playbook_ids(memory, settings)
    registry = default_registry()
    known_tools = {definition.name for definition in registry.list()}
    mined = miner.mine_frequent_detailed(
        sequences,
        support_threshold=support_threshold,
        existing_ids=existing_ids,
        known_tools=known_tools,
    )
    loader = PlaybookLoader(settings.playbooks_path)
    active_examples = _active_trigger_examples(loader)
    for candidate in mined:
        triggers = await _safe_trigger_suggestions(
            memory,
            [conversation_ids[index] for index in candidate.sequence_indexes],
        )
        definition = candidate.definition.model_copy(update={"trigger_examples": triggers})
        level = _required_permission_level(definition, registry)
        adopt_now = (
            auto_adopt
            and level < _ADOPTION_APPROVAL_LEVEL
            and bool(triggers)
            and not _triggers_collide(triggers, active_examples)
        )
        status = "active" if adopt_now else "candidate"
        definition = definition.model_copy(
            update={"status": status, "required_permission_level": level}
        )
        target = settings.playbooks_path / f"{definition.id}.yaml"
        written = guard.write_text(
            target,
            yaml.safe_dump(definition.model_dump(), sort_keys=True),
        )
        guard.validate_table("playbooks")
        await memory.upsert_playbook(
            playbook_id=definition.id,
            name=definition.name,
            source=definition.source,
            status=status,
            trigger_examples=triggers,
            steps=[step.model_dump() for step in definition.steps],
            required_permission_level=level,
            stats=definition.stats.model_dump(),
        )
        report.candidate_ids.append(definition.id)
        report.candidate_paths.append(str(written))
        report.support[definition.id] = candidate.support
        if adopt_now:
            report.adopted_ids.append(definition.id)
            active_examples.extend(triggers)
            report.details.append(
                f"auto-adopted {definition.id} (level {level}, support "
                f"{candidate.support}, {len(definition.steps)} steps)"
            )
        elif level >= _ADOPTION_APPROVAL_LEVEL:
            report.approval_required_ids.append(definition.id)
            report.details.append(
                f"mined {definition.id} with support {candidate.support} "
                f"(level {level} requires adoption approval, status=candidate)"
            )
        elif not triggers:
            report.details.append(
                f"mined {definition.id} with support {candidate.support} "
                "(no safe trigger found; user must provide one before adoption, "
                "status=candidate)"
            )
        else:
            report.details.append(
                f"mined {definition.id} with support {candidate.support} "
                "(trigger is ambiguous with an active playbook, status=candidate)"
            )
    return report


async def _existing_playbook_ids(memory: SqliteMemory, settings: AprilSettings) -> set[str]:
    ids = {playbook.id for playbook in PlaybookLoader(settings.playbooks_path).list()}
    rows = await memory.database.fetchall("SELECT id FROM playbooks")
    ids.update(str(row["id"]) for row in rows)
    return ids


async def _safe_trigger_suggestions(memory: SqliteMemory, conversation_ids: list[str]) -> list[str]:
    """First user message of each supporting conversation, policy-filtered."""
    policy = MemoryPolicy()
    suggestions: list[str] = []
    for conversation_id in conversation_ids:
        row = await memory.database.fetchone(
            """
            SELECT content FROM messages
            WHERE conversation_id = ? AND role = 'user'
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (conversation_id,),
        )
        if row is None:
            continue
        content = " ".join(str(row["content"]).split())
        if not content or len(content) > _MAX_TRIGGER_CHARS:
            continue
        if policy.is_sensitive(content):
            continue
        if content not in suggestions:
            suggestions.append(content)
        if len(suggestions) >= _MAX_TRIGGER_EXAMPLES:
            break
    return suggestions


def _required_permission_level(playbook: PlaybookDefinition, registry: Any) -> int:
    from services.permissions.risk import level_for_risk

    level = 1
    for step in playbook.steps:
        definition = registry.get(step.tool)
        if definition is None:
            # Unknown tools were already filtered during mining; treat any
            # residue as requiring the full approval flow, never auto-adopt.
            return _ADOPTION_APPROVAL_LEVEL
        level = max(level, definition.permission_level, level_for_risk(definition.risk_level))
    return level


def _active_trigger_examples(loader: PlaybookLoader) -> list[str]:
    examples: list[str] = []
    for playbook in loader.list():
        if playbook.status == "active":
            examples.extend(playbook.trigger_examples)
    return examples


def _triggers_collide(candidates: list[str], active_examples: list[str]) -> bool:
    """A suggested trigger colliding with an active playbook's trigger is
    ambiguous: routing would refuse to pick one, so auto-adoption is skipped."""
    normalized_active = {" ".join(example.casefold().split()) for example in active_examples}
    for candidate in candidates:
        normalized = " ".join(candidate.casefold().split())
        for active in normalized_active:
            if normalized in active or active in normalized:
                return True
    return False
