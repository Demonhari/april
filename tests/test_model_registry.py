from __future__ import annotations

from pathlib import Path

import pytest

from april_common.errors import ConfigError
from services.april_runtime.model_lifecycle import ModelLifecycle
from services.april_runtime.model_registry import ModelRegistry


def model_data(path: str = "models/missing.gguf") -> dict[str, object]:
    return {
        "id": "april-brain",
        "name": "granite",
        "path": path,
        "backend": "llama_cpp",
        "role": "brain",
        "threads": 2,
        "context_size": 1024,
        "temperature": 0.2,
        "max_output_tokens": 64,
        "keep_loaded": False,
    }


def test_valid_registry(tmp_path: Path) -> None:
    registry = ModelRegistry.from_dict({"models": {"brain": model_data()}}, root=tmp_path)
    assert registry.get("april-brain").role == "brain"


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    data = {"models": {"one": model_data(), "two": {**model_data(), "name": "other"}}}
    with pytest.raises(ConfigError):
        ModelRegistry.from_dict(data, root=tmp_path)


def test_invalid_values_rejected(tmp_path: Path) -> None:
    data = {"models": {"brain": {**model_data(), "temperature": 3.0}}}
    with pytest.raises(ConfigError):
        ModelRegistry.from_dict(data, root=tmp_path)


def test_missing_model_path_state_is_unavailable(tmp_path: Path) -> None:
    registry = ModelRegistry.from_dict({"models": {"brain": model_data()}}, root=tmp_path)
    lifecycle = ModelLifecycle(registry)
    assert lifecycle.list_models()[0].state == "unavailable"
    assert lifecycle.list_models()[0].missing_path is True
