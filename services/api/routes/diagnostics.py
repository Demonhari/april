from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request

from april_common.errors import AprilError
from april_common.settings import AprilSettings
from services.api.dependencies import ApiContainer
from services.voice.health import voice_health
from services.wake.status import read_wake_status

SettingsPayload = Callable[[AprilSettings], object]
TypedSettingsPayload = Callable[..., object]


def register_diagnostic_routes(
    app: FastAPI,
    authorized: Callable[..., Any],
    *,
    activity_reader: Callable[[Any, int], list[dict[str, Any]]],
    activity_max_limit: int,
    readiness_payload: Callable[[ApiContainer], Awaitable[object]],
    latest_verification_report: TypedSettingsPayload,
    verification_report_history: SettingsPayload,
    verification_report_detail: Callable[[AprilSettings, str], object],
    browser_reports: SettingsPayload,
    browser_latest: TypedSettingsPayload,
    browser_report_types: set[str] | frozenset[str],
) -> None:
    @app.get("/diagnostics")
    async def diagnostics(active: ApiContainer = Depends(authorized)) -> object:
        diagnostic_status = "ok"
        memory_index = await active.memory_repository.health()
        if memory_index.repair_required:
            diagnostic_status = "degraded"
        try:
            runtime = await active.runtime_client.health(timeout=1.0)
            if str(runtime.get("status", "ok")) not in {"ok", "degraded"}:
                diagnostic_status = "degraded"
        except AprilError as exc:
            runtime = {"status": "unavailable", "error": exc.message}
            diagnostic_status = "degraded"
        return {
            "status": diagnostic_status,
            "database": {
                "ok": active.database.path.exists(),
                "path": str(active.database.path),
            },
            "vector_index": active.vector_memory.health(),
            "memory_index": asdict(memory_index),
            "voice": voice_health(active.settings).model_dump(),
            "wake": {
                "enabled": active.settings.wake.enabled,
                "muted": active.settings.mute_flag_path.exists(),
                "state": read_wake_status(active.settings)["state"],
            },
            "scheduler": {
                "enabled": active.settings.scheduler.enabled,
                "running": active.scheduler.running if active.scheduler else False,
                "briefing_enabled": active.settings.scheduler.briefing_enabled,
                "fired_reminders": (
                    active.scheduler.fired_reminder_count if active.scheduler else 0
                ),
            },
            "runtime_url": active.settings.runtime.url,
            "runtime": runtime,
        }

    @app.get("/diagnostics/activity")
    async def diagnostics_activity(
        limit: int = 50,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        capped = max(1, min(limit, activity_max_limit))
        events = activity_reader(active.settings.audit_path, capped)
        return {"events": events, "count": len(events)}

    @app.get("/readiness")
    async def readiness(active: ApiContainer = Depends(authorized)) -> object:
        return await readiness_payload(active)

    @app.get("/verification/report/latest")
    async def verification_report_latest(
        type: str = "any",
        active: ApiContainer = Depends(authorized),
    ) -> object:
        return latest_verification_report(active.settings, report_type=type)

    @app.get("/verification/reports")
    async def verification_reports(
        request: Request,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        if request.query_params:
            raise HTTPException(status_code=400, detail="query parameters are not supported")
        return verification_report_history(active.settings)

    @app.get("/verification/reports/{report_basename}")
    async def verification_report_by_basename(
        report_basename: str,
        request: Request,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        if request.query_params:
            raise HTTPException(status_code=400, detail="query parameters are not supported")
        return verification_report_detail(active.settings, report_basename)

    @app.get("/reports")
    async def reports_index(
        request: Request,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        if request.query_params:
            raise HTTPException(status_code=400, detail="query parameters are not supported")
        return browser_reports(active.settings)

    @app.get("/reports/latest")
    async def reports_latest_any(
        request: Request,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        if request.query_params:
            raise HTTPException(status_code=400, detail="query parameters are not supported")
        return browser_latest(active.settings)

    @app.get("/reports/latest/{report_type}")
    async def reports_latest_typed(
        report_type: str,
        request: Request,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        if request.query_params:
            raise HTTPException(status_code=400, detail="query parameters are not supported")
        if report_type not in browser_report_types:
            raise HTTPException(status_code=404, detail="unknown report type")
        return browser_latest(active.settings, report_type=report_type)
