from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from april_common.errors import ConfigError, NotFoundError
from services.april_runtime.schemas import ModelRole


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> Any:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigError(f"Duplicate YAML key: {key}")
        value = loader.construct_object(value_node, deep=deep)
        mapping[key] = value
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


class ModelDefinition(BaseModel):
    VALID_BACKENDS: ClassVar[set[str]] = {"llama_cpp", "fake"}
    VALID_ROLES: ClassVar[set[str]] = {
        "brain",
        "coding",
        "reading",
        "creative",
        "reasoning",
        "system_action",
    }

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    path: Path
    backend: str
    role: ModelRole
    threads: int = Field(gt=0)
    context_size: int = Field(ge=256)
    temperature: float = Field(ge=0, le=2)
    max_output_tokens: int = Field(gt=0)
    keep_loaded: bool = False
    n_gpu_layers: int | None = None
    n_batch: int | None = Field(default=None, gt=0)
    n_ubatch: int | None = Field(default=None, gt=0)
    use_mmap: bool | None = None
    use_mlock: bool | None = None
    chat_format: str | None = None
    idle_unload_seconds: float | None = Field(default=None, gt=0)
    priority: int = 0

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, value: str) -> str:
        if value not in cls.VALID_BACKENDS:
            raise ValueError(f"Unknown backend: {value}")
        return value

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: Path) -> Path:
        if not str(value):
            raise ValueError("model path is required")
        return value

    def resolved_path(self, root: Path) -> Path:
        expanded = self.path.expanduser()
        if expanded.is_absolute():
            return expanded.resolve(strict=False)
        return (root / expanded).resolve(strict=False)


class ModelRegistryConfig(BaseModel):
    models: dict[str, ModelDefinition]

    @model_validator(mode="after")
    def validate_model_ids(self) -> ModelRegistryConfig:
        seen: set[str] = set()
        for model in self.models.values():
            if model.id in seen:
                raise ValueError(f"Duplicate model id: {model.id}")
            seen.add(model.id)
        return self


class ModelRegistry:
    def __init__(self, models: dict[str, ModelDefinition], *, root: Path) -> None:
        self._models_by_id = {model.id: model for model in models.values()}
        self.root = root.resolve()

    @classmethod
    def from_file(cls, path: Path, *, root: Path) -> ModelRegistry:
        try:
            data = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
        except ConfigError:
            raise
        except OSError as exc:
            raise ConfigError(f"Unable to read model registry: {path}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid model registry YAML: {path}") from exc
        try:
            config = ModelRegistryConfig.model_validate(data)
        except ValueError as exc:
            raise ConfigError("Invalid model registry.", {"error": str(exc)}) from exc
        return cls(config.models, root=root)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, root: Path) -> ModelRegistry:
        try:
            config = ModelRegistryConfig.model_validate(data)
        except ValueError as exc:
            raise ConfigError("Invalid model registry.", {"error": str(exc)}) from exc
        return cls(config.models, root=root)

    def get(self, model_id: str) -> ModelDefinition:
        try:
            return self._models_by_id[model_id]
        except KeyError as exc:
            raise NotFoundError("Model", {"model_id": model_id}) from exc

    def list(self) -> builtins.list[ModelDefinition]:
        return builtins.list(self._models_by_id.values())

    def exists(self, model_id: str) -> bool:
        return model_id in self._models_by_id

    def missing_paths(self) -> builtins.list[str]:
        missing: builtins.list[str] = []
        for model in self.list():
            if not model.resolved_path(self.root).exists():
                missing.append(model.id)
        return missing
