from __future__ import annotations

import secrets
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from april_common.errors import (
    AprilError,
    PermissionDeniedError,
    RuntimeUnavailableError,
    error_payload,
)
from april_common.settings import get_settings
from services.april_runtime.health import runtime_health
from services.april_runtime.model_lifecycle import ModelLifecycle
from services.april_runtime.model_registry import ModelRegistry
from services.april_runtime.schemas import (
    CandidateRuntimeRequest,
    CandidateRuntimeResponse,
    ChatRequest,
    EmbedBatchRequest,
    EmbedBatchResponse,
    EmbedRequest,
    EmbedResponse,
    LoadModelRequest,
    ModelOperationResponse,
)
from services.april_runtime.streaming import stream_event


def create_app(lifecycle: ModelLifecycle | None = None) -> FastAPI:
    settings = get_settings()
    if lifecycle is None:
        registry = ModelRegistry.from_file(
            settings.home / "configs" / "models.yaml", root=settings.home
        )
        from services.pool.governor import ResourceGovernor

        active_lifecycle = ModelLifecycle(
            registry,
            root_backend=settings.runtime.backend,
            max_loaded_specialist_models=settings.runtime.max_loaded_specialist_models,
            governor=ResourceGovernor(settings),
        )
    else:
        active_lifecycle = lifecycle

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if settings.runtime.preload_keep_loaded:
            await active_lifecycle.preload()
        try:
            yield
        finally:
            await active_lifecycle.cleanup()

    app = FastAPI(title="April Runtime", version="0.1.0", lifespan=lifespan)
    app.state.lifecycle = active_lifecycle
    app.state.settings = settings

    @app.middleware("http")
    async def enforce_runtime_auth(request: Request, call_next: Any) -> object:
        token = settings.runtime.token
        if token and request.url.path.startswith("/runtime"):
            authorization = request.headers.get("authorization")
            supplied = (
                authorization.removeprefix("Bearer ").strip()
                if authorization and authorization.startswith("Bearer ")
                else None
            )
            if supplied is None or not secrets.compare_digest(supplied, token):
                request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
                error = PermissionDeniedError("Valid Runtime bearer token is required.")
                return JSONResponse(error_payload(error, request_id), status_code=403)
        return await call_next(request)

    @app.exception_handler(AprilError)
    async def april_error_handler(request: Request, exc: AprilError) -> JSONResponse:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        return JSONResponse(error_payload(exc, request_id), status_code=exc.status_code)

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

    @app.post("/runtime/embed")
    async def embed(request: EmbedRequest) -> EmbedResponse:
        request_id = request.request_id or str(uuid.uuid4())
        model_id, vector = await active_lifecycle.embed(request.text, model_id=request.model_id)
        return EmbedResponse(
            request_id=request_id,
            model_id=model_id,
            dimensions=len(vector),
            embedding=vector,
        )

    @app.post("/runtime/embed/batch")
    async def embed_batch(request: EmbedBatchRequest) -> EmbedBatchResponse:
        request_id = request.request_id or str(uuid.uuid4())
        model_id, vectors = await active_lifecycle.embed_many(
            request.texts,
            model_id=request.model_id,
        )
        return EmbedBatchResponse(
            request_id=request_id,
            model_id=model_id,
            count=len(vectors),
            dimensions=len(vectors[0]),
            embeddings=vectors,
            item_indices=list(range(len(vectors))),
        )

    @app.post("/runtime/models/load")
    async def load_model(request: LoadModelRequest) -> ModelOperationResponse:
        request_id = request.request_id or str(uuid.uuid4())
        state = await active_lifecycle.load_model(
            request.model_id, generation_threads=request.generation_threads
        )
        return ModelOperationResponse(
            request_id=request_id,
            model_id=request.model_id,
            state=state.state,
            message="loaded",
            generation_threads=state.loaded_threads,
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
            generation_threads=None,
        )

    @app.post("/runtime/candidates/prepare")
    async def prepare_candidate(request: CandidateRuntimeRequest) -> CandidateRuntimeResponse:
        request_id = request.request_id or str(uuid.uuid4())
        state = await active_lifecycle.prepare_candidate(
            model_id=request.model_id,
            candidate_id=request.candidate_id,
            adapter_path=Path(request.adapter_path),
            adapter_sha256=request.adapter_sha256,
            configuration_sha256=request.configuration_sha256,
            instance_id=request.instance_id,
            load=request.load,
        )
        identity = state.identity
        if identity is None or identity.adapter_sha256 is None:
            raise RuntimeUnavailableError("Candidate identity was not established safely.")
        return CandidateRuntimeResponse(
            request_id=request_id,
            instance_id=identity.instance_id,
            model_id=identity.model_id,
            candidate_id=identity.candidate_id or request.candidate_id,
            base_model_sha256=identity.base_model_sha256,
            adapter_sha256=identity.adapter_sha256,
            configuration_sha256=identity.configuration_sha256,
            state=state.state,
            integrity_state="verified",
            message="candidate_loaded" if request.load else "candidate_prepared",
        )

    @app.post("/runtime/candidates/unload")
    async def unload_candidate(request: CandidateRuntimeRequest) -> CandidateRuntimeResponse:
        request_id = request.request_id or str(uuid.uuid4())
        state = await active_lifecycle.unload_candidate(request.instance_id or "")
        identity = state.identity
        if identity is None or identity.adapter_sha256 is None:
            raise RuntimeUnavailableError("Candidate identity was not established safely.")
        return CandidateRuntimeResponse(
            request_id=request_id,
            instance_id=identity.instance_id,
            model_id=identity.model_id,
            candidate_id=identity.candidate_id or request.candidate_id,
            base_model_sha256=identity.base_model_sha256,
            adapter_sha256=identity.adapter_sha256,
            configuration_sha256=identity.configuration_sha256,
            state=state.state,
            integrity_state="verified",
            message="candidate_unloaded",
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
