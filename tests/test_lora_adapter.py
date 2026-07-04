from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest

from april_common.errors import RuntimeUnavailableError
from services.april_runtime.llama_cpp_backend import LlamaCppBackend
from services.april_runtime.model_registry import ModelDefinition, ModelRegistry


def _model_data(tmp_path: Path, **extra: object) -> dict[str, object]:
    return {
        "id": "april-brain",
        "name": "test-brain",
        "path": str(tmp_path / "brain.gguf"),
        "backend": "llama_cpp",
        "role": "brain",
        "threads": 2,
        "context_size": 512,
        "temperature": 0.3,
        "max_output_tokens": 64,
        **extra,
    }


def test_registry_parses_optional_adapter_path(tmp_path: Path) -> None:
    registry = ModelRegistry.from_dict(
        {
            "models": {
                "brain": _model_data(
                    tmp_path, adapter_path="models/adapters/brain-lora.gguf"
                )
            }
        },
        root=tmp_path,
    )
    model = registry.get("april-brain")
    assert model.adapter_path == Path("models/adapters/brain-lora.gguf")
    resolved = model.resolved_adapter_path(tmp_path)
    assert resolved == tmp_path / "models" / "adapters" / "brain-lora.gguf"


def test_registry_adapter_path_defaults_to_none(tmp_path: Path) -> None:
    registry = ModelRegistry.from_dict(
        {"models": {"brain": _model_data(tmp_path)}}, root=tmp_path
    )
    model = registry.get("april-brain")
    assert model.adapter_path is None
    assert model.resolved_adapter_path(tmp_path) is None


class _RecordingLlama:
    last_kwargs: ClassVar[dict[str, object]] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).last_kwargs = kwargs
        self.metadata: dict[str, str] = {}


@pytest.fixture
def fake_llama_module(monkeypatch: pytest.MonkeyPatch):
    module = types.ModuleType("llama_cpp")
    module.Llama = _RecordingLlama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    _RecordingLlama.last_kwargs = {}
    return module


@pytest.mark.asyncio
async def test_backend_passes_lora_path_when_adapter_exists(
    tmp_path: Path, fake_llama_module
) -> None:
    base = tmp_path / "brain.gguf"
    base.write_bytes(b"GGUF")
    adapter = tmp_path / "adapters" / "brain-lora.gguf"
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b"GGUF")
    model = ModelDefinition.model_validate(
        _model_data(tmp_path, adapter_path=str(adapter))
    )
    backend = LlamaCppBackend()
    await backend.load(model)
    assert _RecordingLlama.last_kwargs["lora_path"] == str(adapter)


@pytest.mark.asyncio
async def test_backend_fails_closed_on_missing_adapter(
    tmp_path: Path, fake_llama_module
) -> None:
    base = tmp_path / "brain.gguf"
    base.write_bytes(b"GGUF")
    model = ModelDefinition.model_validate(
        _model_data(tmp_path, adapter_path=str(tmp_path / "missing-lora.gguf"))
    )
    backend = LlamaCppBackend()
    with pytest.raises(RuntimeUnavailableError) as excinfo:
        await backend.load(model)
    assert "LoRA adapter" in str(excinfo.value)
    # The base model was never loaded without its adapter.
    assert _RecordingLlama.last_kwargs == {}


@pytest.mark.asyncio
async def test_backend_omits_lora_path_without_adapter(
    tmp_path: Path, fake_llama_module
) -> None:
    base = tmp_path / "brain.gguf"
    base.write_bytes(b"GGUF")
    model = ModelDefinition.model_validate(_model_data(tmp_path))
    backend = LlamaCppBackend()
    await backend.load(model)
    assert "lora_path" not in _RecordingLlama.last_kwargs
