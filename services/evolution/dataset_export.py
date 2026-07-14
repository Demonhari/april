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
    preference_pairs: int = 0
    memories: int = 0
    excluded_conversations: int = 0
    details: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "chat_pairs": self.chat_pairs,
            "preference_pairs": self.preference_pairs,
            "memories": self.memories,
            "excluded_conversations": self.excluded_conversations,
        }


@dataclass(frozen=True, slots=True)
class _RatedReply:
    conversation_id: str
    prompt: str
    reply: str
    rating: str
    reply_index: int
    feedback_created_at: str


def _assistant_index_for_run(
    messages: list[dict[str, str]], *, run_created_at: str, feedback_created_at: str
) -> int | None:
    assistants = [
        (index, message)
        for index, message in enumerate(messages)
        if message["role"] == "assistant" and message["created_at"] <= feedback_created_at
    ]
    following = [
        index for index, message in assistants if message["created_at"] >= run_created_at
    ]
    if following:
        # Structured agent runs are recorded before their reply.
        return following[0]
    preceding = [
        index for index, message in assistants if message["created_at"] <= run_created_at
    ]
    # Standard runs are recorded immediately after their reply.
    return preceding[-1] if preceding else None


def _preceding_user(messages: list[dict[str, str]], reply_index: int) -> str | None:
    for index in range(reply_index - 1, -1, -1):
        if messages[index]["role"] == "user":
            return messages[index]["content"]
    return None


def _correction_reply(
    messages: list[dict[str, str]], rejected_index: int, *, feedback_created_at: str
) -> str | None:
    correction_seen = False
    for message in messages[rejected_index + 1 :]:
        if message["role"] == "user":
            if message["created_at"] < feedback_created_at:
                continue
            if correction_seen:
                return None
            correction_seen = True
        elif message["role"] == "assistant" and correction_seen:
            return message["content"]
    return None


async def _rated_replies(
    memory: SqliteMemory,
) -> tuple[list[_RatedReply], dict[str, list[dict[str, str]]]]:
    rows = await memory.database.fetchall(
        """
        SELECT
            feedback_events.id AS feedback_id,
            feedback_events.rating AS rating,
            feedback_events.conversation_id AS feedback_conversation_id,
            feedback_events.created_at AS feedback_created_at,
            agent_runs.conversation_id AS run_conversation_id,
            agent_runs.created_at AS run_created_at
        FROM feedback_events
        JOIN agent_runs ON agent_runs.id = feedback_events.agent_run_id
        WHERE feedback_events.rating IN ('bad', 'good')
          AND agent_runs.conversation_id IS NOT NULL
        ORDER BY feedback_events.created_at, feedback_events.id
        """
    )
    messages_by_conversation: dict[str, list[dict[str, str]]] = {}
    rated: list[_RatedReply] = []
    for row in rows:
        conversation_id = str(row["run_conversation_id"])
        feedback_conversation_id = row["feedback_conversation_id"]
        if (
            feedback_conversation_id is not None
            and str(feedback_conversation_id) != conversation_id
        ):
            continue
        if conversation_id not in messages_by_conversation:
            message_rows = await memory.database.fetchall(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at, rowid
                """,
                (conversation_id,),
            )
            messages_by_conversation[conversation_id] = [
                {
                    "role": str(message["role"]),
                    "content": str(message["content"]),
                    "created_at": str(message["created_at"]),
                }
                for message in message_rows
            ]
        messages = messages_by_conversation[conversation_id]
        reply_index = _assistant_index_for_run(
            messages,
            run_created_at=str(row["run_created_at"]),
            feedback_created_at=str(row["feedback_created_at"]),
        )
        if reply_index is None:
            continue
        prompt = _preceding_user(messages, reply_index)
        if not prompt:
            continue
        rated.append(
            _RatedReply(
                conversation_id=conversation_id,
                prompt=prompt,
                reply=messages[reply_index]["content"],
                rating=str(row["rating"]),
                reply_index=reply_index,
                feedback_created_at=str(row["feedback_created_at"]),
            )
        )
    return rated, messages_by_conversation


async def _preference_rows(
    memory: SqliteMemory, policy: MemoryPolicy
) -> list[dict[str, str]]:
    rated, messages_by_conversation = await _rated_replies(memory)
    good_by_prompt: dict[str, list[_RatedReply]] = {}
    for item in rated:
        if item.rating == "good":
            good_by_prompt.setdefault(item.prompt, []).append(item)

    rows: list[dict[str, str]] = []
    emitted: set[tuple[str, str, str, str]] = set()
    for item in rated:
        if item.rating != "bad":
            continue
        chosen = _correction_reply(
            messages_by_conversation[item.conversation_id],
            item.reply_index,
            feedback_created_at=item.feedback_created_at,
        )
        if chosen is None:
            chosen = next(
                (
                    candidate.reply
                    for candidate in good_by_prompt.get(item.prompt, [])
                    if candidate.reply != item.reply
                ),
                None,
            )
        if not chosen:
            continue
        if any(policy.is_sensitive(text) for text in (item.prompt, chosen, item.reply)):
            continue
        key = (item.conversation_id, item.prompt, chosen, item.reply)
        if key in emitted:
            continue
        emitted.add(key)
        rows.append(
            {
                "type": "preference",
                "conversation_id": item.conversation_id,
                "prompt": item.prompt,
                "chosen": chosen,
                "rejected": item.reply,
            }
        )
    return rows


async def export_finetune_dataset(
    memory: SqliteMemory,
    settings: AprilSettings,
    *,
    guard: EvolutionWriteGuard | None = None,
    dataset_name: str | None = None,
) -> DatasetExportResult:
    """M15 safe scope: export a reviewable JSONL dataset — no training happens.

    Rows are written under data/evolution/datasets (inside the evolution write
    fence). Conversations with any negative feedback remain excluded from chat
    rows, but valid feedback-bound corrections may produce preference rows.
    Deleted/superseded/expired memories and sensitive-looking content are never
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

    preferences = await _preference_rows(memory, policy)
    lines.extend(
        json.dumps(row, sort_keys=True, ensure_ascii=False) for row in preferences
    )

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
        preference_pairs=len(preferences),
        memories=memory_count,
        excluded_conversations=len(negative_conversations),
    )
