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
                "brain": _model_data(tmp_path, adapter_path="models/adapters/brain-lora.gguf")
            }
        },
        root=tmp_path,
    )
    model = registry.get("april-brain")
    assert model.adapter_path == Path("models/adapters/brain-lora.gguf")
    resolved = model.resolved_adapter_path(tmp_path)
    assert resolved == tmp_path / "models" / "adapters" / "brain-lora.gguf"


def test_registry_adapter_path_defaults_to_none(tmp_path: Path) -> None:
    registry = ModelRegistry.from_dict({"models": {"brain": _model_data(tmp_path)}}, root=tmp_path)
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
    model = ModelDefinition.model_validate(_model_data(tmp_path, adapter_path=str(adapter)))
    backend = LlamaCppBackend()
    await backend.load(model)
    assert _RecordingLlama.last_kwargs["lora_path"] == str(adapter)


@pytest.mark.asyncio
async def test_backend_fails_closed_on_missing_adapter(tmp_path: Path, fake_llama_module) -> None:
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
async def test_backend_omits_lora_path_without_adapter(tmp_path: Path, fake_llama_module) -> None:
    base = tmp_path / "brain.gguf"
    base.write_bytes(b"GGUF")
    model = ModelDefinition.model_validate(_model_data(tmp_path))
    backend = LlamaCppBackend()
    await backend.load(model)
    assert "lora_path" not in _RecordingLlama.last_kwargs


@pytest.mark.asyncio
async def test_dataset_export_excludes_rejected_and_unsafe_rows(settings_tmp) -> None:
    """M15 export: bad-feedback conversations, sensitive pairs, and superseded
    memories never reach the fine-tune dataset."""
    import json

    from services.evolution.dataset_export import export_finetune_dataset
    from services.memory.database import Database
    from services.memory.migrations import run_migrations
    from services.memory.sqlite_memory import SqliteMemory

    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    try:
        good = await memory.create_conversation()
        await memory.add_message(good, "user", "plan my day")
        await memory.add_message(good, "assistant", "Here is your plan.")

        rejected = await memory.create_conversation()
        await memory.add_message(rejected, "user", "do the thing")
        await memory.add_message(rejected, "assistant", "Bad answer.")
        await memory.record_feedback_event(rating="bad", reason="wrong", conversation_id=rejected)

        sensitive = await memory.create_conversation()
        await memory.add_message(sensitive, "user", "my password is hunter2")
        await memory.add_message(sensitive, "assistant", "Noted.")

        keeper = await memory.create_memory("I prefer tea", kind="preference", reason="stated")
        await memory.create_memory(
            "I prefer coffee",
            kind="preference",
            reason="old",
            superseded_by=keeper.id,
        )
        await memory.create_memory(
            "my api key is sk-secret-abcdefghijkl",
            kind="fact",
            reason="unsafe",
        )

        result = await export_finetune_dataset(memory, settings_tmp, dataset_name="exclusions")
        assert result.chat_pairs == 1
        assert result.excluded_conversations == 1
        assert result.memories == 1
        rows = [
            json.loads(line)
            for line in result.path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        chat_rows = [row for row in rows if row["type"] == "chat"]
        memory_rows = [row for row in rows if row["type"] == "memory"]
        assert [row["conversation_id"] for row in chat_rows] == [good]
        assert [row["content"] for row in memory_rows] == ["I prefer tea"]
        text = result.path.read_text(encoding="utf-8")
        assert "hunter2" not in text
        assert "sk-secret" not in text
        assert "Bad answer." not in text
    finally:
        await database.close()
