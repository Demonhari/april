from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import FastAPI

from april_common.service_health import probe_service_health
from april_common.settings import AprilSettings, get_settings
from services.api.application import create_application
from services.api.dependencies import ApiContainer, build_container
from services.api.readiness import (
    _model_registry_readiness as _model_registry_readiness,
)
from services.api.readiness import (
    readiness_payload,
)


async def _readiness_payload(active: ApiContainer) -> dict[str, Any]:
    return await readiness_payload(
        active,
        probe_service_health_fn=probe_service_health,
    )


def create_app(container: ApiContainer | None = None) -> FastAPI:
    return create_application(
        container,
        container_builder=build_container,
        readiness_payload=_readiness_payload,
    )


app = create_app()


def main() -> None:
    settings: AprilSettings = get_settings()
    uvicorn.run(
        "services.api.server:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
