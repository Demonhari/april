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
        self,
        prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        self.active_generations += 1
        self.max_active = max(self.max_active, self.active_generations)
        await asyncio.sleep(0.01)
        self.active_generations -= 1
        return GenerationResult(text="ok", input_tokens=1, output_tokens=1)

    async def stream(
        self,
        prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ):
        yield "ok"

    async def tokenize(self, text: str) -> list[int]:
        return text.split()

    async def health(self) -> BackendHealth:
        return BackendHealth(ok=True, message="ok")


class FailingBackend(CountingBackend):
    async def load(self, model: ModelDefinition) -> None:
        raise RuntimeError("load failed")


class OptionCaptureBackend(CountingBackend):
    def __init__(self) -> None:
        super().__init__()
        self.last_top_p: float | None = None
        self.last_stop: list[str] | None = None

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        self.last_top_p = top_p
        self.last_stop = stop
        return await super().generate(
            prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            stop=stop,
            seed=seed,
        )


class SlowBackend(CountingBackend):
    async def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        self.active_generations += 1
        self.max_active = max(self.max_active, self.active_generations)
        await asyncio.sleep(0.2)
        self.active_generations -= 1
        return GenerationResult(text="ok", input_tokens=1, output_tokens=1)


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


def multi_registry(tmp_path: Path) -> ModelRegistry:
    base = {
        "name": "fake",
        "path": "missing.gguf",
        "backend": "fake",
        "chat_format": "generic",
        "threads": 1,
        "context_size": 1024,
        "temperature": 0.2,
        "max_output_tokens": 64,
    }
    return ModelRegistry.from_dict(
        {
            "models": {
                "brain": {
                    **base,
                    "id": "april-brain",
                    "role": "brain",
                    "keep_loaded": True,
                    "priority": 100,
                },
                "coding": {
                    **base,
                    "id": "april-coding",
                    "role": "coding",
                    "keep_loaded": False,
                    "idle_unload_seconds": 0.01,
                    "priority": 10,
                },
                "reading": {
                    **base,
                    "id": "april-reading",
                    "role": "reading",
                    "keep_loaded": False,
                    "priority": 5,
                },
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


@pytest.mark.asyncio
async def test_generation_options_reach_backend(tmp_path: Path) -> None:
    backend = OptionCaptureBackend()
    lifecycle = ModelLifecycle(
        registry(tmp_path), backend_factory=lambda model: backend, root_backend="fake"
    )
    request = ChatRequest(
        model_id="april-brain",
        messages=[ChatMessage(role="user", content="hello")],
        options={"top_p": 0.7, "stop": ["END"]},
    )
    await lifecycle.generate(request)
    assert backend.last_top_p == 0.7
    assert backend.last_stop == ["END"]


@pytest.mark.asyncio
async def test_active_model_cannot_unload(tmp_path: Path) -> None:
    backend = SlowBackend()
    lifecycle = ModelLifecycle(
        registry(tmp_path), backend_factory=lambda model: backend, root_backend="fake"
    )
    request = ChatRequest(
        model_id="april-brain",
        messages=[ChatMessage(role="user", content="hello")],
    )
    task = asyncio.create_task(lifecycle.generate(request))
    await asyncio.sleep(0.05)
    with pytest.raises(ModelUnavailableError):
        await lifecycle.unload_model("april-brain")
    await task


@pytest.mark.asyncio
async def test_idle_unload_unloads_specialist_but_not_keep_loaded(tmp_path: Path) -> None:
    backends: dict[str, CountingBackend] = {}

    def factory(model: ModelDefinition) -> CountingBackend:
        backend = CountingBackend()
        backends[model.id] = backend
        return backend

    lifecycle = ModelLifecycle(
        multi_registry(tmp_path),
        backend_factory=factory,
        root_backend="fake",
        max_loaded_specialist_models=2,
    )
    await lifecycle.load_model("april-brain")
    await lifecycle.load_model("april-coding")
    await asyncio.sleep(0.02)
    await lifecycle.load_model("april-reading")
    states = {model.id: model.state for model in lifecycle.list_models()}
    assert states["april-brain"] == "loaded"
    assert states["april-coding"] == "unloaded"
    assert states["april-reading"] == "loaded"
    assert backends["april-coding"].unloads == 1


@pytest.mark.asyncio
async def test_lru_evicts_lowest_priority_specialist(tmp_path: Path) -> None:
    lifecycle = ModelLifecycle(
        multi_registry(tmp_path),
        backend_factory=lambda model: CountingBackend(),
        root_backend="fake",
        max_loaded_specialist_models=1,
    )
    await lifecycle.load_model("april-coding")
    await lifecycle.load_model("april-reading")
    states = {model.id: model.state for model in lifecycle.list_models()}
    assert states["april-coding"] == "unloaded"
    assert states["april-reading"] == "loaded"


@pytest.mark.asyncio
async def test_active_specialist_cannot_be_evicted(tmp_path: Path) -> None:
    lifecycle = ModelLifecycle(
        multi_registry(tmp_path),
        backend_factory=lambda model: CountingBackend(),
        root_backend="fake",
        max_loaded_specialist_models=1,
    )
    await lifecycle.load_model("april-coding")
    lifecycle.get_state("april-coding").active_requests = 1
    with pytest.raises(ModelUnavailableError):
        await lifecycle.load_model("april-reading")
