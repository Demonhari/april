"""Model-registry production readiness checks used by the Core API."""

from __future__ import annotations

from typing import Any

from april_common.errors import AprilError
from april_common.model_artifacts import gguf_artifact_status
from april_common.settings import AprilSettings
from services.april_runtime.model_registry import ModelRegistry


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
            else gguf_artifact_status(model.resolved_path(registry.root))
        )
        for model in required_models
    }
    unavailable = sorted(
        model_id
        for model_id, status in artifact_statuses.items()
        if status not in {"valid", "simulated"}
    )
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
            or router_model.resolved_path(registry.root).is_file()
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
            and settings.runtime.backend == "llama_cpp"
            and all(status == "valid" for status in artifact_statuses.values())
        ),
        "reasoning_model_ids": [model.id for model in registry.list() if model.role == "reasoning"],
        "router_model_id": router_model_id,
        "router_aliased_to_brain": router_aliased,
        "dedicated_router_available": dedicated_router_available,
        "router_failure_reason": router_failure_reason,
    }
