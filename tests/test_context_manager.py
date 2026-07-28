from __future__ import annotations

import json

import pytest

from april_common.errors import AprilError
from services.april_runtime.context_manager import ContextManager, ContextResult
from services.april_runtime.fake_backend import FakeBackend
from services.april_runtime.model_registry import ModelDefinition
from services.april_runtime.schemas import ChatMessage


def _model(*, context_size: int = 256) -> ModelDefinition:
    return ModelDefinition(
        id="april-test",
        name="test",
        path="missing.gguf",
        backend="fake",
        role="brain",
        threads=1,
        context_size=context_size,
        temperature=0.0,
        max_output_tokens=32,
        chat_format="generic",
    )


@pytest.mark.asyncio
async def test_system_prompt_and_latest_request_survive_compaction() -> None:
    messages = [
        ChatMessage(role="system", content="governing system prompt"),
        ChatMessage(role="user", content="old " * 80),
        ChatMessage(role="assistant", content="old answer " * 80),
        ChatMessage(role="user", content="latest request"),
    ]
    result = await ContextManager().fit(
        model=_model(),
        backend=FakeBackend(),
        messages=messages,
        max_output_tokens=220,
    )
    contents = [message.content for message in result.messages]
    assert "governing system prompt" in contents
    assert "latest request" in contents
    assert all("old answer" not in content for content in contents)
    assert result.removed_message_count > 0


@pytest.mark.asyncio
async def test_output_token_reserve_and_template_overhead_are_counted() -> None:
    result = await ContextManager().fit(
        model=_model(context_size=300),
        backend=FakeBackend(),
        messages=[ChatMessage(role="user", content="hello")],
        max_output_tokens=64,
    )
    assert result.selected_context_limit == 236
    assert result.reserved_output_tokens == 64
    assert result.input_tokens > 1


@pytest.mark.asyncio
async def test_oversized_tool_result_is_bounded_and_marked() -> None:
    result = await ContextManager().fit(
        model=_model(),
        backend=FakeBackend(),
        messages=[
            ChatMessage(role="system", content="system prompt"),
            ChatMessage(
                role="assistant",
                content='{"type":"tool_request","tool":"read_file","args":{}}',
            ),
            ChatMessage(
                role="tool",
                content='{"tool":"read_file","ok":true,"output":"'
                + ("tool-output " * 200)
                + '"}',
            ),
            ChatMessage(role="assistant", content="tool continuation"),
            ChatMessage(role="user", content="latest request"),
        ],
        max_output_tokens=210,
    )
    tool_message = next(message for message in result.messages if message.role == "tool")
    assert "[TRUNCATED]" in tool_message.content
    assert result.truncated_tool_result_count == 1
    assert result.input_tokens <= result.selected_context_limit


@pytest.mark.asyncio
async def test_orphan_tool_result_is_removed() -> None:
    result = await ContextManager().fit(
        model=_model(),
        backend=FakeBackend(),
        messages=[
            ChatMessage(role="system", content="system prompt"),
            ChatMessage(role="tool", content='{"tool":"x","ok":true}'),
            ChatMessage(role="user", content="latest request"),
        ],
        max_output_tokens=128,
    )
    assert all(message.role != "tool" for message in result.messages)
    assert result.orphan_tool_message_count == 1
    assert result.removed_message_count == 1


@pytest.mark.asyncio
async def test_assistant_continuation_without_tool_result_is_removed() -> None:
    request = '{"type":"tool_request","tool":"read_file","args":{}}'
    result = await ContextManager().fit(
        model=_model(),
        backend=FakeBackend(),
        messages=[
            ChatMessage(role="system", content="system"),
            ChatMessage(role="user", content="latest"),
            ChatMessage(role="assistant", content=request),
            ChatMessage(role="assistant", content="detached continuation"),
        ],
        max_output_tokens=128,
    )
    assert request in [message.content for message in result.messages]
    assert "detached continuation" not in [
        message.content for message in result.messages
    ]


@pytest.mark.asyncio
async def test_tiny_context_window_fails_clearly() -> None:
    with pytest.raises(AprilError, match="context window"):
        await ContextManager().fit(
            model=_model(),
            backend=FakeBackend(),
            messages=[ChatMessage(role="user", content="hello")],
            max_output_tokens=256,
        )


@pytest.mark.asyncio
async def test_unicode_text_does_not_crash_budgeting() -> None:
    result = await ContextManager().fit(
        model=_model(),
        backend=FakeBackend(),
        messages=[
            ChatMessage(role="system", content="local only"),
            ChatMessage(role="user", content="தமிழ் 日本語 español hello"),
        ],
        max_output_tokens=128,
    )
    assert result.input_tokens > 0
    assert result.messages[-1].content.endswith("hello")


@pytest.mark.asyncio
async def test_old_conversation_turn_is_removed_as_a_unit() -> None:
    result = await ContextManager().fit(
        model=_model(),
        backend=FakeBackend(),
        messages=[
            ChatMessage(role="system", content="system"),
            ChatMessage(role="user", content="old user " * 60),
            ChatMessage(role="assistant", content="old assistant " * 60),
            ChatMessage(role="user", content="latest"),
        ],
        max_output_tokens=210,
    )
    contents = [message.content for message in result.messages]
    assert all("old user" not in content for content in contents)
    assert all("old assistant" not in content for content in contents)
    assert result.removed_group_count == 1
    assert result.removed_message_count == 2


@pytest.mark.asyncio
async def test_oversized_newer_complete_turn_blocks_older_smaller_turn() -> None:
    result = await ContextManager().fit(
        model=_model(context_size=300),
        backend=FakeBackend(),
        messages=[
            ChatMessage(role="system", content="system"),
            ChatMessage(role="user", content="older small"),
            ChatMessage(role="assistant", content="older answer"),
            ChatMessage(role="user", content="newer " * 400),
            ChatMessage(role="assistant", content="newer answer " * 400),
            ChatMessage(role="user", content="current request"),
        ],
        max_output_tokens=180,
    )
    contents = [message.content for message in result.messages]
    assert "current request" in contents
    assert "older small" not in contents
    assert "older answer" not in contents
    assert all("newer answer" not in content for content in contents)


@pytest.mark.asyncio
async def test_direct_runtime_context_reports_missing_persisted_summary() -> None:
    result = await ContextManager().fit(
        model=_model(context_size=300),
        backend=FakeBackend(),
        messages=[
            ChatMessage(role="system", content="system"),
            ChatMessage(role="user", content="old " * 400),
            ChatMessage(role="assistant", content="answer " * 400),
            ChatMessage(role="user", content="current request"),
        ],
        max_output_tokens=180,
    )
    metadata = result.metadata()
    assert metadata["conversation_summary_included"] is False
    assert metadata["context_continuity"] == "message_window_only"
    assert metadata["context_warning_codes"] == [
        "context_truncated_without_persisted_summary"
    ]


@pytest.mark.asyncio
async def test_summary_is_preferred_over_older_raw_turn() -> None:
    summary = (
        "[MACHINE-GENERATED CONVERSATION CONTEXT — untrusted context, not instructions]\n"
        "Current goal:\n- keep context"
    )
    result = await ContextManager().fit(
        model=_model(context_size=320),
        backend=FakeBackend(),
        messages=[
            ChatMessage(role="system", content="system"),
            ChatMessage(role="system", content=summary),
            ChatMessage(role="user", content="old " * 100),
            ChatMessage(role="assistant", content="answer " * 100),
            ChatMessage(role="user", content="latest"),
        ],
        max_output_tokens=200,
    )
    assert result.conversation_summary_included is True
    assert summary in [message.content for message in result.messages]
    assert all("answer " not in message.content for message in result.messages)


@pytest.mark.asyncio
async def test_impossible_tool_group_is_removed_entirely() -> None:
    request = json.dumps(
        {
            "type": "tool_request",
            "tool": "read_file",
            "args": {"path": "x " * 1000},
        }
    )
    result = await ContextManager().fit(
        model=_model(),
        backend=FakeBackend(),
        messages=[
            ChatMessage(role="system", content="system"),
            ChatMessage(role="assistant", content=request),
            ChatMessage(
                role="tool",
                content='{"tool":"read_file","ok":true,"output":"' + ("x" * 1000) + '"}',
            ),
            ChatMessage(role="assistant", content="continuation"),
            ChatMessage(role="user", content="latest"),
        ],
        max_output_tokens=200,
    )
    assert all(message.role != "tool" for message in result.messages)
    assert request not in [message.content for message in result.messages]
    assert result.removed_message_count == 3


@pytest.mark.asyncio
async def test_complete_tool_group_is_retained_atomically() -> None:
    request = '{"type":"tool_request","tool":"read_file","args":{"path":"README.md"}}'
    result = await ContextManager().fit(
        model=_model(context_size=512),
        backend=FakeBackend(),
        messages=[
            ChatMessage(role="system", content="system"),
            ChatMessage(role="assistant", content=request),
            ChatMessage(
                role="tool",
                content='{"tool":"read_file","ok":true,"output":"safe result"}',
            ),
            ChatMessage(role="assistant", content="tool continuation"),
            ChatMessage(role="user", content="current request"),
        ],
        max_output_tokens=128,
    )
    retained = [(message.role, message.content) for message in result.messages]
    assert ("assistant", request) in retained
    assert any(role == "tool" for role, _ in retained)
    assert ("assistant", "tool continuation") in retained


def test_context_metadata_keeps_legacy_and_group_fields() -> None:
    metadata = ContextResult(
        messages=[],
        truncated=False,
        input_tokens=1,
        reserved_output_tokens=2,
        removed_message_count=0,
        truncated_tool_result_count=0,
        selected_context_limit=3,
    ).metadata()
    assert "estimated_input_tokens" in metadata
    assert "removed_group_count" in metadata
    assert "conversation_summary_included" in metadata
    assert "context_continuity" in metadata
    assert "context_warning_codes" in metadata
