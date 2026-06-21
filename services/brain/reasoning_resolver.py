from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from services.april_runtime.client import RuntimeClient

logger = logging.getLogger(__name__)

# Runtime model states that mean the runtime can currently serve the model.
# In real mode a missing model file resolves to "unavailable"; in fake mode
# every registered model resolves to "unloaded". Both behaviours are honoured
# here so the resolver only upgrades to a model the runtime can actually run.
_AVAILABLE_STATES: frozenset[str] = frozenset({"unloaded", "loading", "loaded"})
_REASONING_ROLE = "reasoning"


@dataclass(frozen=True, slots=True)
class ReasoningModelResolution:
    model_id: str
    requested_role: str
    selected_role: str
    reason: str
    fallback_model_id: str

    def metadata(self) -> dict[str, str]:
        return {
            "requested_role": self.requested_role,
            "selected_role": self.selected_role,
            "selected_model_id": self.model_id,
            "fallback_model_id": self.fallback_model_id,
            "reason": self.reason,
        }


async def resolve_reasoning_model_id(
    *,
    runtime_client: RuntimeClient,
    fallback_model_id: str,
) -> str:
    resolution = await resolve_reasoning_model(
        runtime_client=runtime_client,
        fallback_model_id=fallback_model_id,
    )
    return resolution.model_id


async def resolve_reasoning_model(
    *,
    runtime_client: RuntimeClient,
    fallback_model_id: str,
) -> ReasoningModelResolution:
    """Return the model id the reasoning agent should use for a run.

    If the runtime currently reports an available model whose role is
    ``reasoning``, its id is returned. Otherwise the agent's configured
    fallback model id is returned. This function never raises: on any
    runtime or registry error it logs and returns ``fallback_model_id``.
    """

    try:
        payload = await runtime_client.models()
    except Exception:
        # Fail safe to the configured brain model on any runtime/registry error.
        logger.warning(
            "Reasoning model resolution failed to list runtime models; falling back to %s.",
            fallback_model_id,
            exc_info=True,
        )
        return ReasoningModelResolution(
            model_id=fallback_model_id,
            requested_role=_REASONING_ROLE,
            selected_role="brain",
            fallback_model_id=fallback_model_id,
            reason="runtime_model_listing_failed",
        )

    upgrade = _first_available_reasoning_model_id(payload)
    if upgrade is not None:
        return ReasoningModelResolution(
            model_id=upgrade,
            requested_role=_REASONING_ROLE,
            selected_role=_REASONING_ROLE,
            fallback_model_id=fallback_model_id,
            reason="available_reasoning_model",
        )
    return ReasoningModelResolution(
        model_id=fallback_model_id,
        requested_role=_REASONING_ROLE,
        selected_role="brain",
        fallback_model_id=fallback_model_id,
        reason="no_available_reasoning_model",
    )


def _first_available_reasoning_model_id(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    models = payload.get("models")
    if not isinstance(models, list):
        return None
    for entry in models:
        if not isinstance(entry, dict):
            continue
        if entry.get("role") != _REASONING_ROLE:
            continue
        if entry.get("state") not in _AVAILABLE_STATES:
            continue
        model_id = entry.get("id")
        if isinstance(model_id, str) and model_id:
            return model_id
    return None
