from __future__ import annotations

import pytest

from services.april_runtime.llama_cpp_backend import LlamaCppBackend, llama_response_format
from services.april_runtime.schemas import ChatMessage, ResponseFormat


class FakeLlama:
    def __init__(self, *, fail_chat: bool = False, reject_response_format: bool = False) -> None:
        self.fail_chat = fail_chat
        self.reject_response_format = reject_response_format
        self.chat_calls: list[dict[str, object]] = []
        self.prompt_calls: list[str] = []

    def create_chat_completion(self, *, messages: list[dict[str, str]], **kwargs: object) -> object:
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        if self.reject_response_format and "response_format" in kwargs:
            # Mimics an older llama build that does not accept the argument.
            raise TypeError("response_format is not supported by this build")
        if self.fail_chat:
            raise RuntimeError("chat unsupported")
        if kwargs.get("stream"):
            return iter(
                [
                    {"choices": [{"delta": {"content": "hello "}}]},
                    {"choices": [{"delta": {"content": "world"}}]},
                ]
            )
        return {
            "choices": [{"message": {"content": "chat response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }

    def __call__(self, prompt: str, **kwargs: object) -> object:
        self.prompt_calls.append(prompt)
        if kwargs.get("stream"):
            return iter(
                [
                    {"choices": [{"text": "fallback "}]},
                    {"choices": [{"text": "stream"}]},
                ]
            )
        return {"choices": [{"text": "fallback response", "finish_reason": "stop"}]}

    def tokenize(self, text: bytes) -> list[int]:
        return [index for index, _part in enumerate(text.decode("utf-8").split())]


def backend_with(llm: FakeLlama) -> LlamaCppBackend:
    backend = LlamaCppBackend()
    backend._llm = llm
    return backend


@pytest.mark.asyncio
async def test_llama_backend_prefers_chat_completion() -> None:
    llm = FakeLlama()
    backend = backend_with(llm)
    result = await backend.generate_messages(
        "USER: hello\nASSISTANT:",
        messages=[ChatMessage(role="user", content="hello")],
        temperature=0.0,
        max_output_tokens=8,
    )
    assert result.text == "chat response"
    assert result.input_tokens == 5
    assert result.output_tokens == 2
    assert llm.chat_calls
    assert not llm.prompt_calls


@pytest.mark.asyncio
async def test_llama_backend_falls_back_to_prompt_completion() -> None:
    llm = FakeLlama(fail_chat=True)
    backend = backend_with(llm)
    result = await backend.generate_messages(
        "USER: hello\nASSISTANT:",
        messages=[ChatMessage(role="user", content="hello")],
        temperature=0.0,
        max_output_tokens=8,
    )
    assert result.text == "fallback response"
    assert llm.prompt_calls == ["USER: hello\nASSISTANT:"]
    assert result.input_tokens > 0
    assert result.output_tokens > 0


@pytest.mark.asyncio
async def test_llama_backend_streams_chat_completion() -> None:
    llm = FakeLlama()
    backend = backend_with(llm)
    tokens = [
        token
        async for token in backend.stream_messages(
            "USER: hello\nASSISTANT:",
            messages=[ChatMessage(role="user", content="hello")],
            temperature=0.0,
            max_output_tokens=8,
        )
    ]
    assert tokens == ["hello ", "world"]
    assert llm.chat_calls


@pytest.mark.asyncio
async def test_llama_backend_stream_falls_back_safely() -> None:
    llm = FakeLlama(fail_chat=True)
    backend = backend_with(llm)
    tokens = [
        token
        async for token in backend.stream_messages(
            "USER: hello\nASSISTANT:",
            messages=[ChatMessage(role="user", content="hello")],
            temperature=0.0,
            max_output_tokens=8,
        )
    ]
    assert tokens == ["fallback ", "stream"]
    assert llm.prompt_calls == ["USER: hello\nASSISTANT:"]


def test_llama_backend_extracts_only_prompt_metadata_keys() -> None:
    # A loaded Llama exposes a metadata mapping; only the prompt-rendering keys
    # are retained, and they are returned by value (never the live mapping).
    llm = FakeLlama()
    llm.metadata = {  # type: ignore[attr-defined]
        "tokenizer.chat_template": "{% for m in messages %}[{{ m.role }}]{% endfor %}",
        "general.architecture": "granite",
        "tokenizer.ggml.model": "gpt2",
    }
    backend = LlamaCppBackend()
    from services.april_runtime.llama_cpp_backend import _extract_prompt_metadata

    extracted = _extract_prompt_metadata(llm)
    assert extracted == {
        "tokenizer.chat_template": "{% for m in messages %}[{{ m.role }}]{% endfor %}"
    }
    assert "general.architecture" not in extracted
    # Default backend (no loaded model) reports no metadata.
    assert backend.prompt_metadata() == {}


def test_llama_backend_metadata_missing_is_safe() -> None:
    # A llama-cpp-python build without a metadata mapping must yield {}.
    from services.april_runtime.llama_cpp_backend import _extract_prompt_metadata

    class _NoMetadata:
        pass

    assert _extract_prompt_metadata(_NoMetadata()) == {}


def test_llama_response_format_translation() -> None:
    assert llama_response_format(None) is None
    assert llama_response_format(ResponseFormat(type="text")) is None
    assert llama_response_format(ResponseFormat(type="json_object")) == {"type": "json_object"}
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert llama_response_format(ResponseFormat(type="json_object", json_schema=schema)) == {
        "type": "json_object",
        "schema": schema,
    }


@pytest.mark.asyncio
async def test_llama_passes_response_format_to_chat_completion() -> None:
    llm = FakeLlama()
    backend = backend_with(llm)
    schema = {"type": "object", "properties": {"intent": {"type": "string"}}}
    await backend.generate_messages(
        "USER: hello\nASSISTANT:",
        messages=[ChatMessage(role="user", content="hello")],
        temperature=0.0,
        max_output_tokens=8,
        response_format=ResponseFormat(type="json_object", json_schema=schema),
    )
    assert llm.chat_calls[0]["kwargs"]["response_format"] == {  # type: ignore[index]
        "type": "json_object",
        "schema": schema,
    }


@pytest.mark.asyncio
async def test_llama_streams_with_response_format() -> None:
    llm = FakeLlama()
    backend = backend_with(llm)
    tokens = [
        token
        async for token in backend.stream_messages(
            "USER: hello\nASSISTANT:",
            messages=[ChatMessage(role="user", content="hello")],
            temperature=0.0,
            max_output_tokens=8,
            response_format=ResponseFormat(type="json_object"),
        )
    ]
    assert tokens == ["hello ", "world"]
    assert llm.chat_calls[0]["kwargs"]["response_format"] == {"type": "json_object"}  # type: ignore[index]


@pytest.mark.asyncio
async def test_llama_degrades_when_response_format_unsupported() -> None:
    # An older backend rejects response_format; generation must still succeed by
    # degrading to prompt completion rather than crashing.
    llm = FakeLlama(reject_response_format=True)
    backend = backend_with(llm)
    result = await backend.generate_messages(
        "USER: hello\nASSISTANT:",
        messages=[ChatMessage(role="user", content="hello")],
        temperature=0.0,
        max_output_tokens=8,
        response_format=ResponseFormat(type="json_object"),
    )
    assert result.text == "fallback response"
    assert llm.prompt_calls == ["USER: hello\nASSISTANT:"]
