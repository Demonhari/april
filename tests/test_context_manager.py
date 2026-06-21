from __future__ import annotations

import pytest

from april_common.errors import AprilError
from services.april_runtime.context_manager import ContextManager
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
            ChatMessage(role="tool", content="tool-output " * 200),
            ChatMessage(role="user", content="latest request"),
        ],
        max_output_tokens=210,
    )
    tool_message = next(message for message in result.messages if message.role == "tool")
    assert "[TRUNCATED]" in tool_message.content
    assert result.truncated_tool_result_count == 1
    assert result.input_tokens <= result.selected_context_limit


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
