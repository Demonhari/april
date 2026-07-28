from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from agents.registry import AgentRegistry, default_agent_registry
from april_common.audit import AuditLogger
from april_common.settings import ConversationContextSettings
from services.april_runtime.schemas import ChatMessage, ChatResponse, Usage
from services.brain.conversation_context import (
    ConversationContextService,
    group_persisted_conversation_turns,
)
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.schemas import ConversationSummaryContent, Message
from services.memory.sqlite_memory import SqliteMemory


class SummaryRuntime:
    def __init__(self, responses: list[str], *, delay: float = 0.0) -> None:
        self.responses = responses
        self.delay = delay
        self.calls: list[list[ChatMessage]] = []
        self.options: list[Any] = []

    async def chat(
        self,
        *,
        model_id: str,
        messages: list[ChatMessage],
        options: Any = None,
        response_format: Any = None,
        request_id: str | None = None,
    ) -> ChatResponse:
        self.calls.append(messages)
        self.options.append(options)
        if self.delay:
            await asyncio.sleep(self.delay)
        return ChatResponse(
            request_id=request_id or "summary",
            model_id=model_id,
            content=self.responses.pop(0),
            usage=Usage(),
        )


async def _memory(path: Path) -> tuple[Database, SqliteMemory, str]:
    database = Database(path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    conversation_id = await memory.create_conversation()
    return database, memory, conversation_id


async def _add_turns(
    memory: SqliteMemory, conversation_id: str, count: int, *, start: int = 0
) -> None:
    for index in range(start, start + count):
        user_id = await memory.add_message(conversation_id, "user", f"user {index}")
        assistant_id = await memory.add_message(
            conversation_id, "assistant", f"assistant {index}"
        )
        for message_id, suffix in ((user_id, "0"), (assistant_id, "1")):
            await memory.database.execute(
                "UPDATE messages SET created_at = ? WHERE id = ?",
                (f"2026-01-01T00:00:{index:02d}.{suffix}Z", message_id),
            )


def _summary_json(goal: str = "Ship the change") -> str:
    return json.dumps(
        {
            "current_goal": goal,
            "important_facts": ["APRIL is local-first"],
            "decisions": [],
            "constraints": ["Do not use cloud APIs"],
            "completed_actions": [],
            "open_loops": ["Run tests"],
        }
    )


@pytest.mark.asyncio
async def test_short_conversation_makes_no_summary_call(settings_tmp: Any) -> None:
    database, memory, conversation_id = await _memory(settings_tmp.database_path)
    await _add_turns(memory, conversation_id, 4)
    runtime = SummaryRuntime([])
    service = ConversationContextService(
        memory=memory,
        runtime_client=runtime,  # type: ignore[arg-type]
        agent_registry=default_agent_registry(),
        settings=ConversationContextSettings(),
    )
    prepared = await service.prepare(conversation_id=conversation_id)
    assert runtime.calls == []
    assert prepared.summary is None
    assert prepared.recent_turn_count == 4
    assert next(item.content for item in prepared.recent_messages) == "user 0"
    await database.close()


@pytest.mark.asyncio
async def test_incremental_summary_advances_once_and_keeps_four_turns(
    settings_tmp: Any,
) -> None:
    database, memory, conversation_id = await _memory(settings_tmp.database_path)
    await _add_turns(memory, conversation_id, 7)
    runtime = SummaryRuntime([_summary_json()])
    service = ConversationContextService(
        memory=memory,
        runtime_client=runtime,  # type: ignore[arg-type]
        agent_registry=default_agent_registry(),
        settings=ConversationContextSettings(),
        audit=AuditLogger(settings_tmp.audit_path),
    )
    prepared = await service.prepare(conversation_id=conversation_id, request_id="r")
    assert len(runtime.calls) == 1
    assert prepared.summary_advanced is True
    assert prepared.summary_version == 1
    assert prepared.recent_turn_count == 4
    assert next(item.content for item in prepared.recent_messages) == "user 3"
    summary = await memory.get_conversation_summary(conversation_id)
    assert summary is not None
    assert summary.summarized_message_count == 6
    assert summary.through_message_id == (
        await memory.list_messages_paginated(conversation_id, limit=6)
    )[-1].id

    again = await service.prepare(conversation_id=conversation_id, request_id="r2")
    assert len(runtime.calls) == 1
    assert again.summary_advanced is False
    audit_text = settings_tmp.audit_path.read_text(encoding="utf-8")
    assert "Ship the change" not in audit_text
    assert "assistant 0" not in audit_text
    await database.close()


@pytest.mark.asyncio
async def test_previous_summary_is_incremental_input(settings_tmp: Any) -> None:
    database, memory, conversation_id = await _memory(settings_tmp.database_path)
    await _add_turns(memory, conversation_id, 7)
    runtime = SummaryRuntime([_summary_json("First"), _summary_json("Second")])
    service = ConversationContextService(
        memory=memory,
        runtime_client=runtime,  # type: ignore[arg-type]
        agent_registry=default_agent_registry(),
        settings=ConversationContextSettings(),
    )
    await service.prepare(conversation_id=conversation_id)
    await _add_turns(memory, conversation_id, 3, start=7)
    await service.prepare(conversation_id=conversation_id)
    assert len(runtime.calls) == 2
    second_payload = json.loads(runtime.calls[1][1].content)
    assert second_payload["previous_summary"]["current_goal"] == "First"
    assert "user 0" not in runtime.calls[1][1].content
    await database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("response", ["not json", '{"current_goal":"x","extra":1}'])
async def test_invalid_summary_leaves_checkpoint_unchanged(
    settings_tmp: Any, response: str
) -> None:
    database, memory, conversation_id = await _memory(settings_tmp.database_path)
    await _add_turns(memory, conversation_id, 7)
    service = ConversationContextService(
        memory=memory,
        runtime_client=SummaryRuntime([response]),  # type: ignore[arg-type]
        agent_registry=default_agent_registry(),
        settings=ConversationContextSettings(),
    )
    prepared = await service.prepare(conversation_id=conversation_id)
    assert prepared.summary_advanced is False
    assert "conversation_summary_unavailable" in prepared.warnings
    assert await memory.get_conversation_summary(conversation_id) is None
    await database.close()


@pytest.mark.asyncio
async def test_timeout_and_missing_reading_model_degrade_safely(
    settings_tmp: Any,
) -> None:
    database, memory, conversation_id = await _memory(settings_tmp.database_path)
    await _add_turns(memory, conversation_id, 7)
    settings = ConversationContextSettings(summary_timeout_seconds=0.01)
    timed_out = ConversationContextService(
        memory=memory,
        runtime_client=SummaryRuntime([_summary_json()], delay=0.1),  # type: ignore[arg-type]
        agent_registry=default_agent_registry(),
        settings=settings,
    )
    prepared = await timed_out.prepare(conversation_id=conversation_id)
    assert "conversation_summary_unavailable" in prepared.warnings

    missing = ConversationContextService(
        memory=memory,
        runtime_client=SummaryRuntime([]),  # type: ignore[arg-type]
        agent_registry=AgentRegistry([]),
        settings=settings,
    )
    prepared = await missing.prepare(conversation_id=conversation_id)
    assert "conversation_summary_unavailable" in prepared.warnings
    await database.close()


@pytest.mark.asyncio
async def test_secret_like_summary_items_are_not_persisted(settings_tmp: Any) -> None:
    database, memory, conversation_id = await _memory(settings_tmp.database_path)
    await _add_turns(memory, conversation_id, 7)
    secret_response = json.dumps(
        {
            "current_goal": "password=do-not-store",
            "important_facts": [
                "API_KEY=do-not-store",
                "raw tool output: full private file contents",
                "safe fact",
            ],
            "decisions": [],
            "constraints": [],
            "completed_actions": [],
            "open_loops": [],
        }
    )
    service = ConversationContextService(
        memory=memory,
        runtime_client=SummaryRuntime([secret_response]),  # type: ignore[arg-type]
        agent_registry=default_agent_registry(),
        settings=ConversationContextSettings(),
    )
    await service.prepare(conversation_id=conversation_id)
    summary = await memory.get_conversation_summary(conversation_id)
    assert summary is not None
    assert summary.content.current_goal is None
    assert summary.content.important_facts == ["safe fact"]
    await database.close()


@pytest.mark.asyncio
async def test_incomplete_newest_turn_is_never_summarized(settings_tmp: Any) -> None:
    database, memory, conversation_id = await _memory(settings_tmp.database_path)
    await _add_turns(memory, conversation_id, 7)
    pending_id = await memory.add_message(conversation_id, "user", "pending approval turn")
    await memory.database.execute(
        "UPDATE messages SET created_at = ? WHERE id = ?",
        ("2026-01-01T00:00:08Z", pending_id),
    )
    runtime = SummaryRuntime([_summary_json()])
    service = ConversationContextService(
        memory=memory,
        runtime_client=runtime,  # type: ignore[arg-type]
        agent_registry=default_agent_registry(),
        settings=ConversationContextSettings(),
    )
    prepared = await service.prepare(conversation_id=conversation_id)
    assert len(runtime.calls) == 1
    assert "pending approval turn" not in runtime.calls[0][1].content
    assert prepared.recent_messages[-1].content == "pending approval turn"
    summary = await memory.get_conversation_summary(conversation_id)
    assert summary is not None
    assert summary.through_message_id != pending_id
    await database.close()


@pytest.mark.asyncio
async def test_recent_bound_preserves_current_request_and_drops_older_after_oversize(
    settings_tmp: Any,
) -> None:
    database, memory, conversation_id = await _memory(settings_tmp.database_path)
    await memory.add_message(conversation_id, "user", "older small")
    await memory.add_message(conversation_id, "assistant", "older answer")
    await memory.add_message(conversation_id, "user", "newer " * 200)
    await memory.add_message(conversation_id, "assistant", "newer answer " * 200)
    await memory.add_message(conversation_id, "user", "current request")
    service = ConversationContextService(
        memory=memory,
        runtime_client=SummaryRuntime([]),  # type: ignore[arg-type]
        agent_registry=default_agent_registry(),
        settings=ConversationContextSettings(
            summary_enabled=False,
            conversation_history_max_chars=1000,
        ),
    )
    prepared = await service.prepare(conversation_id=conversation_id)
    assert [message.content for message in prepared.recent_messages] == [
        "current request"
    ]
    await database.close()


def test_grouping_preserves_complete_turns_and_reports_orphans() -> None:
    def message(index: int, role: str, content: str) -> Message:
        return Message(
            id=str(index),
            conversation_id="c",
            role=role,
            content=content,
            created_at=f"2026-01-01T00:00:0{index}Z",
        )

    groups, warnings = group_persisted_conversation_turns(
        [
            message(0, "assistant", "orphan"),
            message(1, "tool", "orphan tool"),
            message(2, "user", "first"),
            message(3, "assistant", "answer"),
            message(4, "user", "unfinished"),
        ]
    )
    assert groups[2].complete is True
    assert [item.role for item in groups[2].messages] == ["user", "assistant"]
    assert groups[3].complete is False
    assert "orphan_assistant_message" in warnings
    assert "orphan_tool_message" in warnings


@pytest.mark.asyncio
async def test_summary_cas_and_cascade(settings_tmp: Any) -> None:
    database, memory, conversation_id = await _memory(settings_tmp.database_path)
    await _add_turns(memory, conversation_id, 2)
    messages = await memory.list_messages_paginated(conversation_id, limit=10)
    first = await memory.upsert_conversation_summary(
        conversation_id=conversation_id,
        content=ConversationSummaryContent(current_goal="first"),
        through_message_id=messages[1].id,
        through_created_at=messages[1].created_at,
        summarized_message_count=2,
        source_hash="a" * 64,
        model_id="april-reading",
        expected_version=None,
    )
    assert first is not None
    assert first.version == 1
    raw = await database.fetchone(
        "SELECT summary_json FROM conversation_summaries WHERE conversation_id = ?",
        (conversation_id,),
    )
    assert raw is not None
    assert raw["summary_json"] == json.dumps(
        first.content.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    stale = await memory.upsert_conversation_summary(
        conversation_id=conversation_id,
        content=ConversationSummaryContent(current_goal="stale"),
        through_message_id=messages[3].id,
        through_created_at=messages[3].created_at,
        summarized_message_count=4,
        source_hash="b" * 64,
        model_id="april-reading",
        expected_version=0,
    )
    backwards = await memory.upsert_conversation_summary(
        conversation_id=conversation_id,
        content=ConversationSummaryContent(current_goal="backwards"),
        through_message_id=messages[0].id,
        through_created_at=messages[0].created_at,
        summarized_message_count=1,
        source_hash="c" * 64,
        model_id="april-reading",
        expected_version=1,
        expected_through_message_id=first.through_message_id,
        expected_source_hash=first.source_hash,
    )
    assert stale is None
    assert backwards is None
    assert (await memory.get_conversation_summary(conversation_id)).version == 1  # type: ignore[union-attr]
    await memory.delete_conversation(conversation_id)
    assert await memory.get_conversation_summary(conversation_id) is None
    await database.close()
