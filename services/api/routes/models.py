from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import Depends, FastAPI

from services.api.dependencies import ApiContainer
from services.april_runtime.schemas import LoadModelRequest
from services.pool.agent_pool import AgentPool


def register_model_routes(
    app: FastAPI,
    authorized: Callable[..., Any],
) -> None:
    @app.get("/pool/agents")
    async def pool_agents(active: ApiContainer = Depends(authorized)) -> object:
        pool = AgentPool(
            active.memory,
            known_agents=[agent.name for agent in active.agent_registry.list()],
        )
        return {"agents": [card.to_payload() for card in await pool.scorecards()]}

    @app.get("/runtime/models")
    async def runtime_models(active: ApiContainer = Depends(authorized)) -> object:
        return await active.runtime_client.models()

    @app.post("/runtime/models/load")
    async def runtime_model_load(
        request: LoadModelRequest,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        return await active.runtime_client.load(request.model_id, request_id=request.request_id)

    @app.post("/runtime/models/unload")
    async def runtime_model_unload(
        request: LoadModelRequest,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        return await active.runtime_client.unload(request.model_id, request_id=request.request_id)
