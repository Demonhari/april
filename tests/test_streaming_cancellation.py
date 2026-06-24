from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

from services.april_runtime.fake_backend import FakeBackend
from services.april_runtime.model_lifecycle import ModelLifecycle
from services.april_runtime.model_registry import ModelRegistry
from services.april_runtime.schemas import ChatMessage, ChatRequest
from tests.test_llama_cpp_backend import FakeLlama, backend_with
from tests.test_runtime_api import runtime_lifecycle


def _brain_request(content: str = "plan my work today") -> ChatRequest:
    return ChatRequest(model_id="april-brain", messages=[ChatMessage(role="user", content=content)])


async def _wait_event(event: threading.Event, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not event.is_set():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("background work did not finish in time")
        await asyncio.sleep(0.01)


async def test_active_requests_returns_to_zero_after_cancel(tmp_path: Path) -> None:
    lifecycle = runtime_lifecycle(tmp_path)
    gen = lifecycle.stream(_brain_request())
    await gen.__anext__()  # meta
    await gen.__anext__()  # first token
    assert lifecycle.get_state("april-brain").active_requests == 1
    await asyncio.wait_for(gen.aclose(), timeout=2.0)
    assert lifecycle.get_state("april-brain").active_requests == 0
    await lifecycle.cleanup()


async def test_cancelled_stream_does_not_poison_next_request(tmp_path: Path) -> None:
    lifecycle = runtime_lifecycle(tmp_path)
    gen = lifecycle.stream(_brain_request())
    await gen.__anext__()  # meta
    await gen.__anext__()  # one token
    await asyncio.wait_for(gen.aclose(), timeout=2.0)
    assert lifecycle.get_state("april-brain").active_requests == 0

    # A brand-new stream after a cancellation must complete normally.
    events = [event async for event in lifecycle.stream(_brain_request())]
    names = [name for name, _ in events]
    assert names[0] == "meta"
    assert "token" in names
    assert names[-1] == "done"
    assert lifecycle.get_state("april-brain").active_requests == 0
    await lifecycle.cleanup()


async def test_stream_exception_is_converted_to_error_event(tmp_path: Path) -> None:
    registry = ModelRegistry.from_dict(
        {
            "models": {
                "brain": {
                    "id": "april-brain",
                    "name": "fake",
                    "path": "missing.gguf",
                    "backend": "fake",
                    "role": "brain",
                    "chat_format": "generic",
                    "threads": 1,
                    "context_size": 1024,
                    "temperature": 0.2,
                    "max_output_tokens": 64,
                    "keep_loaded": False,
                }
            }
        },
        root=tmp_path,
    )
    lifecycle = ModelLifecycle(
        registry,
        root_backend="fake",
        backend_factory=lambda model: FakeBackend(fail_stream=True),
    )
    events = [event async for event in lifecycle.stream(_brain_request("hi"))]
    error_events = [payload for name, payload in events if name == "error"]
    assert error_events
    assert error_events[0]["code"] == "GENERATION_FAILED"
    assert lifecycle.get_state("april-brain").active_requests == 0
    await lifecycle.cleanup()


async def test_runtime_shutdown_after_active_stream_is_clean(tmp_path: Path) -> None:
    lifecycle = runtime_lifecycle(tmp_path)
    gen = lifecycle.stream(_brain_request())
    await gen.__anext__()  # meta
    await gen.__anext__()  # token
    # Simulates the server cancelling an in-flight request during shutdown.
    await asyncio.wait_for(gen.aclose(), timeout=2.0)
    assert lifecycle.get_state("april-brain").active_requests == 0
    # Shutdown unload must not hang or fail now that the request is released.
    await asyncio.wait_for(lifecycle.cleanup(), timeout=2.0)
    assert lifecycle.get_state("april-brain").state in {"unloaded", "unavailable"}


class _InfiniteLlama(FakeLlama):
    def __init__(self) -> None:
        super().__init__()
        self.generator_closed = threading.Event()

    def create_chat_completion(self, *, messages: list[dict[str, str]], **kwargs: object) -> object:
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})

        def _endless() -> Any:
            index = 0
            try:
                while True:
                    yield {"choices": [{"delta": {"content": f"t{index} "}}]}
                    index += 1
            finally:
                self.generator_closed.set()

        return _endless()


async def test_backend_stream_cancellation_closes_underlying_generator() -> None:
    llm = _InfiniteLlama()
    backend = backend_with(llm)
    gen = backend.stream_messages(
        "USER: hi\nASSISTANT:",
        messages=[ChatMessage(role="user", content="hi")],
        temperature=0.0,
        max_output_tokens=8,
    )
    assert await gen.__anext__()  # first token
    await asyncio.wait_for(gen.aclose(), timeout=2.0)
    # The llama generator's finally must run, proving generation actually stopped.
    await _wait_event(llm.generator_closed)
