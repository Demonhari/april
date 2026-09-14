"""Model-registry production readiness checks used by the Core API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from april_common.errors import AprilError
from april_common.model_artifacts import gguf_artifact_status
from april_common.settings import AprilSettings
from services.april_runtime.colibri_backend import validate_colibri_url
from services.april_runtime.model_registry import ModelDefinition, ModelRegistry


def _artifact_ready(model: ModelDefinition, path: Any) -> bool:
    backend = model.backend
    if backend == "colibri":
        if not path.is_dir():
            return False
        if any(
            ".." in Path(relative).parts or Path(relative).is_absolute()
            for relative in model.colibri_expected_files
        ):
            return False
        if not all((path / relative).is_file() for relative in model.colibri_expected_files):
            return False
        tokenizer = model.colibri_tokenizer_path
        if tokenizer is not None and not tokenizer.is_absolute():
            tokenizer = path / tokenizer
        return tokenizer is not None and tokenizer.is_file() and model.resident_gb is not None
    return path.is_file()


def model_registry_readiness(settings: AprilSettings) -> dict[str, Any]:
    router_model_id = settings.brain.router_model_id or settings.brain.model_id
    router_aliased = settings.brain.router_model_id is None
    try:
        registry = ModelRegistry.from_file(
            settings.home / "configs" / "models.yaml",
            root=settings.home,
        )
    except AprilError:
        return {
            "valid": False,
            "required_model_available": False,
            "required_model_ids": [],
            "unavailable_required_model_ids": [],
            "production_required_roles": ["brain", "coding", "reading"],
            "missing_production_required_roles": ["brain", "coding", "reading"],
            "artifact_statuses": {},
            "production_model_artifacts_ready": False,
            "reasoning_model_ids": [],
            "router_model_id": router_model_id,
            "router_aliased_to_brain": router_aliased,
            "dedicated_router_available": False,
            "router_failure_reason": "model_registry_invalid",
        }
    production_required_roles = {"brain", "coding", "reading"}
    required_models = [
        model for model in registry.list() if model.role in production_required_roles
    ]
    registered_required_roles = {model.role for model in required_models}
    missing_required_roles = sorted(production_required_roles - registered_required_roles)
    artifact_statuses = {
        model.id: (
            "simulated"
            if settings.runtime.backend == "fake" or model.backend == "fake"
            else (
                gguf_artifact_status(model.resolved_path(registry.root))
                if model.backend == "llama_cpp"
                else (
                    "valid"
                    if _artifact_ready(model, model.resolved_path(registry.root))
                    else "missing"
                )
            )
        )
        for model in required_models
    }
    unavailable = sorted(
        model_id
        for model_id, status in artifact_statuses.items()
        if status not in {"valid", "simulated"}
    )
    colibri_endpoint_ready = all(
        model.backend != "colibri" or _valid_colibri_endpoint(model.colibri_base_url)
        for model in required_models
    )
    colibri_readiness = {
        model.id: {
            "tokenizer_configured": (
                model.colibri_tokenizer_path is not None
                and (
                    model.resolved_path(registry.root) / model.colibri_tokenizer_path
                    if model.colibri_tokenizer_path is not None
                    and not model.colibri_tokenizer_path.is_absolute()
                    else model.colibri_tokenizer_path
                ).is_file()
            ),
            "resident_estimate_configured": model.resident_gb is not None,
            "endpoint_configured": _valid_colibri_endpoint(model.colibri_base_url),
        }
        for model in required_models
        if model.backend == "colibri"
    }
    router_failure_reason: str | None = None
    dedicated_router_available = False
    if router_aliased:
        router_valid = registry.exists(settings.brain.model_id)
        if not router_valid:
            router_failure_reason = "aliased_brain_model_not_registered"
    elif not registry.exists(router_model_id):
        router_valid = False
        router_failure_reason = "dedicated_router_not_registered"
    else:
        router_model = registry.get(router_model_id)
        router_valid = router_model.role == "router"
        dedicated_router_available = router_valid and (
            settings.runtime.backend == "fake"
            or router_model.backend == "fake"
            or _artifact_ready(router_model, router_model.resolved_path(registry.root))
        )
        if not router_valid:
            router_failure_reason = "dedicated_router_role_mismatch"
        elif not dedicated_router_available:
            router_failure_reason = "dedicated_router_artifact_unavailable"
    return {
        "valid": True,
        "required_model_available": (bool(required_models) and not unavailable and router_valid),
        "required_model_ids": [model.id for model in required_models],
        "unavailable_required_model_ids": unavailable,
        "production_required_roles": sorted(production_required_roles),
        "missing_production_required_roles": missing_required_roles,
        "artifact_statuses": artifact_statuses,
        "production_model_artifacts_ready": bool(
            not missing_required_roles
            and not unavailable
            and settings.runtime.backend in {"llama_cpp", "colibri"}
            and all(status == "valid" for status in artifact_statuses.values())
            and colibri_endpoint_ready
        ),
        "colibri_readiness": colibri_readiness,
        "reasoning_model_ids": [model.id for model in registry.list() if model.role == "reasoning"],
        "router_model_id": router_model_id,
        "router_aliased_to_brain": router_aliased,
        "dedicated_router_available": dedicated_router_available,
        "router_failure_reason": router_failure_reason,
    }


def _valid_colibri_endpoint(value: str | None) -> bool:
    try:
        validate_colibri_url(value)
    except ValueError:
        return False
    return True
