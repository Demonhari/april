from __future__ import annotations

import os
from pathlib import Path

from apps.runner.mac_report import quantization_from_basename
from apps.runner.multi_model_report import PerModelResult
from apps.runner.verification.types import ModelPlanEntry
from services.april_runtime.model_registry import ModelRegistry


def plan_multi_model_verification(home: Path, *, llama_available: bool) -> list[ModelPlanEntry]:
    """Plan local real-model checks without downloading or loading artifacts."""
    registry = ModelRegistry.from_file(home / "configs" / "models.yaml", root=home)
    entries: list[ModelPlanEntry] = []
    for model in registry.list():
        path = model.resolved_path(registry.root)
        if model.backend != "llama_cpp":
            reason = f"Backend {model.backend} is not a real GGUF backend."
            entries.append(ModelPlanEntry(model=model, path=path, available=False, reason=reason))
        elif model.role == "embedding":
            entries.append(
                ModelPlanEntry(
                    model=model,
                    path=path,
                    available=False,
                    reason="Embedding model is verified via `run april memory reindex`, not chat.",
                )
            )
        elif not llama_available:
            entries.append(
                ModelPlanEntry(
                    model=model,
                    path=path,
                    available=False,
                    reason="llama-cpp-python is not installed (pip install -e '.[runtime]').",
                )
            )
        elif not path.exists():
            entries.append(
                ModelPlanEntry(
                    model=model,
                    path=path,
                    available=False,
                    reason=f"Missing model file: {path}",
                )
            )
        elif not os.access(path, os.R_OK):
            entries.append(
                ModelPlanEntry(
                    model=model,
                    path=path,
                    available=False,
                    reason=f"Not readable: {path}",
                )
            )
        else:
            entries.append(ModelPlanEntry(model=model, path=path, available=True, reason=None))
    return entries


def skipped_result_for(entry: ModelPlanEntry) -> PerModelResult:
    """Return a redacted result for a model that was not exercised."""
    return PerModelResult(
        model_id=entry.model.id,
        role=entry.model.role,
        backend=entry.model.backend,
        path_basename=entry.path_basename,
        quantization=quantization_from_basename(entry.path_basename),
        available=False,
        skipped_reason=entry.reason,
    )
