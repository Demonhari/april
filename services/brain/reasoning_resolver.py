from __future__ import annotations

import logging
from typing import Any

from services.april_runtime.client import RuntimeClient

logger = logging.getLogger(__name__)

# Runtime model states that mean the runtime can currently serve the model.
# In real mode a missing model file resolves to "unavailable"; in fake mode
# every registered model resolves to "unloaded". Both behaviours are honoured
# here so the resolver only upgrades to a model the runtime can actually run.
_AVAILABLE_STATES: frozenset[str] = frozenset(
    {"unloaded", "loading", "loaded"}
)
_REASONING_ROLE = "reasoning"


async def resolve_reasoning_model_id(
    *,
    runtime_client: RuntimeClient,
    fallback_model_id: str,
) -> str:
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
            "Reasoning model resolution failed to list runtime models; "
            "falling back to %s.",
            fallback_model_id,
            exc_info=True,
        )
        return fallback_model_id

    upgrade = _first_available_reasoning_model_id(payload)
    if upgrade is not None:
        return upgrade
    return fallback_model_id


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
