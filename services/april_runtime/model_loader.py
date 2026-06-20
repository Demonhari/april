from __future__ import annotations

from services.april_runtime.model_lifecycle import ModelLifecycle, ModelRuntimeState


class ModelLoader:
    """Thin lifecycle facade kept for the public runtime module contract."""

    def __init__(self, lifecycle: ModelLifecycle) -> None:
        self.lifecycle = lifecycle

    async def load(self, model_id: str) -> ModelRuntimeState:
        return await self.lifecycle.load_model(model_id)

    async def unload(self, model_id: str) -> ModelRuntimeState:
        return await self.lifecycle.unload_model(model_id)
