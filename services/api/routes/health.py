from __future__ import annotations

from fastapi import FastAPI


def register_health_routes(app: FastAPI) -> None:
    @app.get("/health")
    async def health() -> object:
        return {"status": "ok", "service": "april-core-api"}
