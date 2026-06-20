from __future__ import annotations

from services.april_runtime.model_registry import ModelDefinition
from services.april_runtime.schemas import GenerationOptions


def effective_generation_options(
    model: ModelDefinition,
    overrides: GenerationOptions,
) -> tuple[float, int]:
    temperature = model.temperature if overrides.temperature is None else overrides.temperature
    max_output_tokens = (
        model.max_output_tokens
        if overrides.max_output_tokens is None
        else min(overrides.max_output_tokens, model.max_output_tokens)
    )
    return temperature, max_output_tokens
