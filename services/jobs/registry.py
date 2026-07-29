from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class EmptyPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RepositoryIndexPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_path: str = Field(min_length=1, max_length=4096)
    project_id: str | None = Field(default=None, max_length=128)
    force_full_reindex: bool = False


class MemoryReindexPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentIndexPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    folder_path: str = Field(min_length=1, max_length=4096)


class ConfiguredTestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    argv: list[str] = Field(min_length=1, max_length=32)
    cwd: str = Field(min_length=1, max_length=4096)


class ModelVerificationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1, max_length=128)


class BenchmarkPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1, max_length=128)


class FinetunePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(min_length=16, max_length=64, pattern=r"^[a-f0-9]+$")


@dataclass(frozen=True, slots=True)
class JobTypeDefinition:
    name: str
    payload_model: type[BaseModel]
    permission_level: int
    approval_required: bool
    idempotent: bool
    restart_safe: bool
    default_timeout_seconds: float
    maximum_attempts: int
    cancellation_behavior: str
    available: bool
    unavailable_code: str | None = None

    def validate_payload(self, value: dict[str, Any]) -> dict[str, Any]:
        validated = TypeAdapter(self.payload_model).validate_python(value)
        return validated.model_dump(mode="json")


class JobRegistry:
    def __init__(self, definitions: list[JobTypeDefinition]) -> None:
        self._definitions = {definition.name: definition for definition in definitions}
        if len(self._definitions) != len(definitions):
            raise ValueError("Duplicate job type definition.")

    def get(self, name: str) -> JobTypeDefinition | None:
        return self._definitions.get(name)

    def require(self, name: str) -> JobTypeDefinition:
        definition = self.get(name)
        if definition is None:
            raise ValueError("Unknown job type.")
        if not definition.available:
            raise ValueError(definition.unavailable_code or "job_type_unavailable")
        return definition

    def definitions(self) -> tuple[JobTypeDefinition, ...]:
        return tuple(self._definitions.values())


def default_job_registry(
    *,
    finetune_enabled: bool = False,
    evolution_enabled: bool = False,
) -> JobRegistry:
    def safe(
        name: str,
        payload_model: type[BaseModel],
    ) -> JobTypeDefinition:
        return JobTypeDefinition(
            name=name,
            payload_model=payload_model,
            permission_level=2,
            approval_required=False,
            idempotent=True,
            restart_safe=True,
            default_timeout_seconds=1800.0,
            maximum_attempts=2,
            cancellation_behavior="cooperative",
            available=True,
        )

    return JobRegistry(
        [
            safe("self_check", EmptyPayload),
            safe("repository_index", RepositoryIndexPayload),
            safe("memory_reindex", MemoryReindexPayload),
            safe("document_index", DocumentIndexPayload),
            JobTypeDefinition(
                "configured_test",
                ConfiguredTestPayload,
                permission_level=3,
                approval_required=True,
                idempotent=True,
                restart_safe=True,
                default_timeout_seconds=1800.0,
                maximum_attempts=2,
                cancellation_behavior="process_group",
                available=True,
            ),
            JobTypeDefinition(
                "model_import_verification",
                ModelVerificationPayload,
                permission_level=2,
                approval_required=False,
                idempotent=True,
                restart_safe=True,
                default_timeout_seconds=900.0,
                maximum_attempts=2,
                cancellation_behavior="process_group",
                available=True,
            ),
            JobTypeDefinition(
                "model_benchmark",
                BenchmarkPayload,
                permission_level=2,
                approval_required=False,
                idempotent=True,
                restart_safe=True,
                default_timeout_seconds=3600.0,
                maximum_attempts=2,
                cancellation_behavior="process_group",
                available=True,
            ),
            JobTypeDefinition(
                "finetune",
                FinetunePayload,
                permission_level=4,
                approval_required=True,
                idempotent=True,
                restart_safe=False,
                default_timeout_seconds=172_800.0,
                maximum_attempts=2,
                cancellation_behavior="process_group",
                available=finetune_enabled,
                unavailable_code="finetune_job_disabled",
            ),
            JobTypeDefinition(
                "dream_cycle",
                EmptyPayload,
                permission_level=2,
                approval_required=False,
                idempotent=False,
                restart_safe=False,
                default_timeout_seconds=1800.0,
                maximum_attempts=1,
                cancellation_behavior="cooperative",
                available=evolution_enabled,
                unavailable_code="dream_cycle_job_unavailable",
            ),
        ]
    )
