from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from april_common.errors import AprilError, error_payload
from april_common.settings import get_settings
from services.april_runtime.health import runtime_health
from services.april_runtime.model_lifecycle import ModelLifecycle
from services.april_runtime.model_registry import ModelRegistry
from services.april_runtime.schemas import ChatRequest, LoadModelRequest, ModelOperationResponse
from services.april_runtime.streaming import stream_event


def create_app(lifecycle: ModelLifecycle | None = None) -> FastAPI:
    settings = get_settings()
    if lifecycle is None:
        registry = ModelRegistry.from_file(
            settings.home / "configs" / "models.yaml", root=settings.home
        )
        active_lifecycle = ModelLifecycle(
            registry,
            root_backend=settings.runtime.backend,
        )
    else:
        active_lifecycle = lifecycle
    app = FastAPI(title="April Runtime", version="0.1.0")
    app.state.lifecycle = active_lifecycle
    app.state.settings = settings

    @app.exception_handler(AprilError)
    async def april_error_handler(request: Request, exc: AprilError) -> JSONResponse:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        return JSONResponse(error_payload(exc, request_id), status_code=exc.status_code)

    @app.on_event("startup")
    async def startup() -> None:
        if settings.runtime.preload_keep_loaded:
            await active_lifecycle.preload()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await active_lifecycle.cleanup()

    @app.post("/runtime/chat")
    async def chat(request: ChatRequest) -> object:
        return await active_lifecycle.generate(request)

    @app.post("/runtime/stream")
    async def stream(request: ChatRequest) -> StreamingResponse:
        request_id = request.request_id or str(uuid.uuid4())

        async def events() -> AsyncIterator[str]:
            async for event_name, payload in active_lifecycle.stream(
                request.model_copy(update={"request_id": request_id})
            ):
                yield stream_event(
                    event=event_name,
                    request_id=request_id,
                    model_id=request.model_id,
                    payload=payload if isinstance(payload, dict) else {},
                )

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/runtime/models/load")
    async def load_model(request: LoadModelRequest) -> ModelOperationResponse:
        request_id = request.request_id or str(uuid.uuid4())
        state = await active_lifecycle.load_model(request.model_id)
        return ModelOperationResponse(
            request_id=request_id,
            model_id=request.model_id,
            state=state.state,
            message="loaded",
        )

    @app.post("/runtime/models/unload")
    async def unload_model(request: LoadModelRequest) -> ModelOperationResponse:
        request_id = request.request_id or str(uuid.uuid4())
        state = await active_lifecycle.unload_model(request.model_id)
        return ModelOperationResponse(
            request_id=request_id,
            model_id=request.model_id,
            state=state.state,
            message="unloaded",
        )

    @app.get("/runtime/models")
    async def models() -> object:
        return {"models": active_lifecycle.list_models()}

    @app.get("/runtime/health")
    async def health() -> object:
        return runtime_health(
            active_lifecycle,
            backend=settings.runtime.backend,
        )

    return app


app = create_app()


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "services.april_runtime.server:app",
        host=settings.runtime.host,
        port=settings.runtime.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
