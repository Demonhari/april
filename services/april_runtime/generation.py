from __future__ import annotations

from dataclasses import dataclass

from services.april_runtime.model_registry import ModelDefinition
from services.april_runtime.schemas import GenerationOptions


@dataclass(frozen=True, slots=True)
class EffectiveGenerationOptions:
    temperature: float
    max_output_tokens: int
    top_p: float | None
    stop: list[str]
    seed: int | None


def effective_generation_options(
    model: ModelDefinition,
    overrides: GenerationOptions,
) -> EffectiveGenerationOptions:
    temperature = model.temperature if overrides.temperature is None else overrides.temperature
    max_output_tokens = (
        model.max_output_tokens
        if overrides.max_output_tokens is None
        else min(overrides.max_output_tokens, model.max_output_tokens)
    )
    return EffectiveGenerationOptions(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=overrides.top_p,
        stop=list(overrides.stop),
        seed=overrides.seed,
    )
