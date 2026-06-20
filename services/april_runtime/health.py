from __future__ import annotations

import uuid

from services.april_runtime.model_lifecycle import ModelLifecycle
from services.april_runtime.schemas import RuntimeHealth


def runtime_health(
    lifecycle: ModelLifecycle, *, backend: str, request_id: str | None = None
) -> RuntimeHealth:
    models = lifecycle.list_models()
    missing = [model.id for model in models if model.missing_path]
    loaded = [model for model in models if model.state == "loaded"]
    return RuntimeHealth(
        status="degraded" if missing or any(model.state == "error" for model in models) else "ok",
        backend=backend,
        models=models,
        missing_models=missing,
        request_id=request_id or str(uuid.uuid4()),
        loaded_model_count=len(loaded),
        active_requests=sum(model.active_requests for model in models),
        generation_error_count=sum(model.generation_errors for model in models),
        lifecycle_policy=lifecycle.policy_snapshot(),
    )
