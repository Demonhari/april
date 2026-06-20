from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from april_common.errors import ModelUnavailableError
from services.april_runtime.backend import BackendHealth, GenerationResult, RuntimeBackend
from services.april_runtime.model_lifecycle import ModelLifecycle
from services.april_runtime.model_registry import ModelDefinition, ModelRegistry
from services.april_runtime.schemas import ChatMessage, ChatRequest


class CountingBackend(RuntimeBackend):
    def __init__(self) -> None:
        self.loads = 0
        self.unloads = 0
        self.active_generations = 0
        self.max_active = 0

    async def load(self, model: ModelDefinition) -> None:
        self.loads += 1
        await asyncio.sleep(0.01)

    async def unload(self) -> None:
        self.unloads += 1

    async def generate(
        self, prompt: str, *, temperature: float, max_output_tokens: int
    ) -> GenerationResult:
        self.active_generations += 1
        self.max_active = max(self.max_active, self.active_generations)
        await asyncio.sleep(0.01)
        self.active_generations -= 1
        return GenerationResult(text="ok", input_tokens=1, output_tokens=1)

    async def stream(self, prompt: str, *, temperature: float, max_output_tokens: int):
        yield "ok"

    async def tokenize(self, text: str) -> list[int]:
        return text.split()

    async def health(self) -> BackendHealth:
        return BackendHealth(ok=True, message="ok")


class FailingBackend(CountingBackend):
    async def load(self, model: ModelDefinition) -> None:
        raise RuntimeError("load failed")


def registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry.from_dict(
        {
            "models": {
                "brain": {
                    "id": "april-brain",
                    "name": "fake",
                    "path": "missing.gguf",
                    "backend": "fake",
                    "role": "brain",
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


@pytest.mark.asyncio
async def test_idempotent_load_unload_and_concurrent_load(tmp_path: Path) -> None:
    backend = CountingBackend()
    lifecycle = ModelLifecycle(
        registry(tmp_path), backend_factory=lambda model: backend, root_backend="fake"
    )
    await asyncio.gather(lifecycle.load_model("april-brain"), lifecycle.load_model("april-brain"))
    await lifecycle.load_model("april-brain")
    assert backend.loads == 1
    await lifecycle.unload_model("april-brain")
    await lifecycle.unload_model("april-brain")
    assert backend.unloads == 1


@pytest.mark.asyncio
async def test_generation_lock(tmp_path: Path) -> None:
    backend = CountingBackend()
    lifecycle = ModelLifecycle(
        registry(tmp_path), backend_factory=lambda model: backend, root_backend="fake"
    )
    request = ChatRequest(
        model_id="april-brain",
        messages=[ChatMessage(role="user", content="hello")],
    )
    await asyncio.gather(lifecycle.generate(request), lifecycle.generate(request))
    assert backend.max_active == 1


@pytest.mark.asyncio
async def test_backend_error_state(tmp_path: Path) -> None:
    lifecycle = ModelLifecycle(
        registry(tmp_path),
        backend_factory=lambda model: FailingBackend(),
        root_backend="fake",
    )
    with pytest.raises(ModelUnavailableError):
        await lifecycle.load_model("april-brain")
    assert lifecycle.list_models()[0].state == "error"
