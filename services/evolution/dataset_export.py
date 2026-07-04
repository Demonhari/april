from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from april_common.settings import AprilSettings
from april_common.time import utc_now_iso
from services.evolution.write_guard import EvolutionWriteGuard
from services.memory.policy import MemoryPolicy
from services.memory.sqlite_memory import SqliteMemory


@dataclass(slots=True)
class DatasetExportResult:
    path: Path
    chat_pairs: int = 0
    memories: int = 0
    excluded_conversations: int = 0
    details: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "chat_pairs": self.chat_pairs,
            "memories": self.memories,
            "excluded_conversations": self.excluded_conversations,
        }


async def export_finetune_dataset(
    memory: SqliteMemory,
    settings: AprilSettings,
    *,
    guard: EvolutionWriteGuard | None = None,
    dataset_name: str | None = None,
) -> DatasetExportResult:
    """M15 safe scope: export a reviewable JSONL dataset — no training happens.

    Rows are written under data/evolution/datasets (inside the evolution write
    fence). Conversations with any negative feedback are excluded entirely;
    deleted/superseded/expired memories and sensitive-looking content are never
    exported. The output is meant for manual review before any offline
    fine-tuning run (see scripts/finetune/README.md).
    """
    active_guard = guard or EvolutionWriteGuard(settings)
    policy = MemoryPolicy()
    name = dataset_name or f"dataset-{utc_now_iso()[:10]}"
    target = settings.evolution_path / "datasets" / f"{name}.jsonl"

    negative_rows = await memory.database.fetchall(
        "SELECT DISTINCT conversation_id FROM feedback_events "
        "WHERE rating = 'bad' AND conversation_id IS NOT NULL"
    )
    negative_conversations = {str(row["conversation_id"]) for row in negative_rows}

    lines: list[str] = []
    chat_pairs = 0
    conversations = await memory.database.fetchall(
        "SELECT id FROM conversations ORDER BY created_at"
    )
    for row in conversations:
        conversation_id = str(row["id"])
        if conversation_id in negative_conversations:
            continue
        messages = await memory.recent_messages(conversation_id, limit=200)
        previous_user: str | None = None
        for message in messages:
            if message.role == "user":
                previous_user = message.content
                continue
            if message.role == "assistant" and previous_user:
                if policy.is_sensitive(previous_user) or policy.is_sensitive(message.content):
                    previous_user = None
                    continue
                lines.append(
                    json.dumps(
                        {
                            "type": "chat",
                            "conversation_id": conversation_id,
                            "prompt": previous_user,
                            "response": message.content,
                        },
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                )
                chat_pairs += 1
                previous_user = None

    memory_count = 0
    # list_memories already excludes superseded and expired rows; deleted rows
    # no longer exist. Sensitive-looking content is filtered again here.
    for record in await memory.list_memories():
        if policy.is_sensitive(record.content):
            continue
        lines.append(
            json.dumps(
                {
                    "type": "memory",
                    "memory_id": record.id,
                    "kind": record.kind,
                    "content": record.content,
                    "confidence": record.confidence,
                },
                sort_keys=True,
                ensure_ascii=False,
            )
        )
        memory_count += 1

    written = active_guard.write_text(target, "\n".join(lines) + ("\n" if lines else ""))
    return DatasetExportResult(
        path=written,
        chat_pairs=chat_pairs,
        memories=memory_count,
        excluded_conversations=len(negative_conversations),
    )
