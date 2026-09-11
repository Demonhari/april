from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from april_common.errors import AprilError, RequestTooLargeError, error_payload
from april_common.settings import get_settings
from services.api.activity import ACTIVITY_MAX_LIMIT, read_activity_events
from services.api.auth import require_bearer_token
from services.api.dependencies import ApiContainer
from services.api.reporting import (
    _BROWSER_REPORT_TYPES,
    _browser_latest,
    _browser_reports,
    _latest_verification_report,
    _verification_report_detail,
    _verification_report_history,
)
from services.api.routes.chat import register_chat_routes
from services.api.routes.diagnostics import register_diagnostic_routes
from services.api.routes.evolution import register_evolution_routes
from services.api.routes.health import register_health_routes
from services.api.routes.jobs import register_job_routes
from services.api.routes.memory import register_memory_routes
from services.api.routes.models import register_model_routes
from services.api.routes.projects import register_project_routes
from services.api.routes.scheduler import register_scheduler_routes
from services.api.routes.tools import register_tool_routes
from services.api.routes.voice import register_voice_routes
from services.api.streaming import sse_event
from services.api.wake_events import _handle_wake_event
from services.brain.model_routing import build_routing_request, routing_user_context
from services.brain.structured_output import ROUTING_PROPOSAL_RESPONSE_FORMAT
from services.wake.schemas import WakeEvent
from services.wake.wake_bus import WakeBus

_DESKTOP_WEB_DIR = Path(__file__).resolve().parents[2] / "apps" / "desktop" / "web"


def create_application(
    container: ApiContainer | None = None,
    *,
    container_builder: Callable[[], Awaitable[ApiContainer]],
    readiness_payload: Callable[[ApiContainer], Awaitable[dict[str, Any]]],
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.container is None:
            try:
                app.state.container = await container_builder()
            except Exception:
                # Liveness must remain available when dependency assembly fails.
                # Authenticated endpoints retry assembly and surface the real error.
                app.state.container_error = True
        active: ApiContainer | None = app.state.container
        if active is not None and active.scheduler is not None:
            # start() is a no-op unless scheduler.enabled, so this is safe in tests.
            await active.scheduler.start()
        prewarm_task: asyncio.Task[None] | None = None
        if active is not None and active.settings.runtime.prefix_prewarm:
            prewarm_task = _schedule_router_prewarm(active)
        app.state.router_prewarm_task = prewarm_task
        wake_bus: WakeBus | None = None
        if active is not None and active.settings.wake.enabled:
            # Local wake bus: owner-only Unix socket for hotkey/desktop wakes.
            async def bus_handler(event: WakeEvent) -> dict[str, Any]:
                return await _handle_wake_event(active, event, request_id=str(uuid.uuid4()))

            wake_bus = WakeBus(active.settings.wake_socket_path, bus_handler)
            await wake_bus.start()
        app.state.wake_bus = wake_bus
        yield
        if app.state.wake_bus is not None:
            await app.state.wake_bus.stop()
            app.state.wake_bus = None
        prewarm_task = app.state.router_prewarm_task
        if prewarm_task is not None and not prewarm_task.done():
            prewarm_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await prewarm_task
        app.state.router_prewarm_task = None
        if app.state.container is not None:
            await app.state.container.aclose()

    app = FastAPI(title="APRIL Core API", version="0.1.0", lifespan=lifespan)
    app.state.container = container
    app.state.container_error = False
    app.state.wake_bus = None
    app.state.router_prewarm_task = None
    initial_settings = container.settings if container is not None else get_settings()
    if initial_settings.api.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://127.0.0.1", "http://localhost"],
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        )

    @app.exception_handler(AprilError)
    async def april_error_handler(request: Request, exc: AprilError) -> JSONResponse:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        return JSONResponse(error_payload(exc, request_id), status_code=exc.status_code)

    @app.middleware("http")
    async def enforce_request_size(request: Request, call_next: Any) -> object:
        active_settings = (
            app.state.container.settings if app.state.container is not None else initial_settings
        )
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                length = int(content_length)
            except ValueError:
                length = 0
            if length > active_settings.api.max_request_bytes:
                request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
                error = RequestTooLargeError(
                    "Request body exceeds configured maximum size.",
                    {"max_request_bytes": active_settings.api.max_request_bytes},
                )
                return JSONResponse(error_payload(error, request_id), status_code=413)
        return await call_next(request)

    async def get_container() -> ApiContainer:
        if app.state.container is None:
            app.state.container = await container_builder()
        return app.state.container

    async def authorized(
        authorization: str | None = Header(default=None),
        active: ApiContainer = Depends(get_container),
    ) -> ApiContainer:
        await require_bearer_token(active.settings, authorization)
        return active

    register_health_routes(app)
    register_job_routes(app, authorized)
    register_diagnostic_routes(
        app,
        authorized,
        activity_reader=read_activity_events,
        activity_max_limit=ACTIVITY_MAX_LIMIT,
        readiness_payload=readiness_payload,
        latest_verification_report=_latest_verification_report,
        verification_report_history=_verification_report_history,
        verification_report_detail=_verification_report_detail,
        browser_reports=_browser_reports,
        browser_latest=_browser_latest,
        browser_report_types=_BROWSER_REPORT_TYPES,
    )

    register_chat_routes(app, authorized, sse_event=sse_event)
    register_voice_routes(app, authorized, wake_handler=_handle_wake_event)
    register_tool_routes(app, authorized)
    register_memory_routes(app, authorized)

    register_evolution_routes(app, authorized)
    register_scheduler_routes(app, authorized)
    register_project_routes(app, authorized)
    register_model_routes(app, authorized)

    # Serve the local Desktop SPA from the Core API (same-origin, loopback only).
    # The static assets ship no secrets; all data still flows through the
    # authenticated endpoints above. Mounted last so it never shadows API routes.
    if _DESKTOP_WEB_DIR.is_dir():
        app.mount(
            "/desktop",
            StaticFiles(directory=str(_DESKTOP_WEB_DIR), html=True),
            name="desktop",
        )

    return app


def _schedule_router_prewarm(active: ApiContainer) -> asyncio.Task[None] | None:
    if active.settings.runtime.backend == "fake":
        return None
    task = asyncio.create_task(_router_prewarm(active))
    task.add_done_callback(_consume_prewarm_task)
    return task


async def _router_prewarm(active: ApiContainer) -> None:
    governor = active.governor
    if governor is not None:
        async_method = getattr(governor, "assess_resident_async", None)
        decision = (
            await async_method()
            if callable(async_method)
            else await asyncio.to_thread(governor.assess_resident)
        )
        if not getattr(decision, "allowed", True):
            _write_prewarm_audit(active, "skipped", ",".join(decision.reasons))
            return
    router = active.orchestrator.brain_router
    messages, options = build_routing_request(
        system_prompt=router.router_system_prompt,
        user_content=routing_user_context("ping", None),
        max_output_tokens=1,
    )
    reason: str | None = None
    for attempt in range(3):
        try:
            await active.runtime_client.chat(
                model_id=router.router_model_id,
                messages=messages,
                options=options,
                response_format=ROUTING_PROPOSAL_RESPONSE_FORMAT,
                request_id=f"prewarm-{uuid.uuid4()}",
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reason = type(exc).__name__
            if attempt < 2:
                await asyncio.sleep(10.0 * (attempt + 1))
                continue
            _write_prewarm_audit(active, "failed", reason)
        else:
            _write_prewarm_audit(active, "loaded", None)
        return


def _write_prewarm_audit(active: ApiContainer, status: str, reason: str | None) -> None:
    active.approvals.audit.write(
        {
            "event_type": "router_prefix_prewarm",
            "actor": "core_api",
            "status": status,
            "reason": reason,
        }
    )


def _consume_prewarm_task(task: asyncio.Task[None]) -> None:
    if task.cancelled():
        return
    try:
        task.exception()
    except Exception:
        return
