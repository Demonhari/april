from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from apps.daemon.apriald import _loopback_health_check, default_child_specs
from apps.runner.service_manager import AprilServiceManager
from april_common.service_health import ServiceHealthResult, probe_service_health


@pytest.mark.parametrize(
    ("status_code", "ok", "reason"),
    [
        (200, True, "ok"),
        (204, True, "ok"),
        (302, False, "redirect"),
        (401, False, "authentication_rejected"),
        (403, False, "authentication_rejected"),
        (404, False, "endpoint_not_found"),
        (500, False, "http_error"),
    ],
)
def test_shared_health_status_interpretation(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    ok: bool,
    reason: str,
) -> None:
    request = httpx.Request("GET", "http://127.0.0.1/health")
    monkeypatch.setattr(
        "april_common.service_health.httpx.get",
        lambda *_args, **_kwargs: httpx.Response(status_code, request=request),
    )
    result = probe_service_health(
        "http://127.0.0.1/health",
        bearer_token="top-secret-token",
        timeout=0.25,
    )
    assert result.ok is ok
    assert result.status_code == status_code
    assert result.reason == reason
    assert "top-secret-token" not in result.message


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (httpx.ConnectError("refused"), "connection_failed"),
        (httpx.ReadTimeout("slow"), "timeout"),
    ],
)
def test_shared_health_transport_failures_are_safe(
    monkeypatch: pytest.MonkeyPatch,
    error: httpx.HTTPError,
    reason: str,
) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise error

    monkeypatch.setattr("april_common.service_health.httpx.get", fail)
    result = probe_service_health(
        "http://127.0.0.1/health",
        bearer_token="top-secret-token",
        timeout=0.25,
    )
    assert result.ok is False
    assert result.reason == reason
    assert "top-secret-token" not in result.message


def test_daemon_runtime_probe_uses_endpoint_and_token(
    settings_tmp: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = settings_tmp.model_copy(
        update={
            "runtime": settings_tmp.runtime.model_copy(update={"token": "daemon-runtime-secret"})
        }
    )
    runtime = default_child_specs(settings)[0]
    assert runtime.health_url == f"{settings.runtime.url}/runtime/health"
    assert runtime.health_token == "daemon-runtime-secret"
    captured: dict[str, Any] = {}

    def probe(url: str, *, bearer_token: str | None, timeout: float) -> ServiceHealthResult:
        captured.update(url=url, bearer_token=bearer_token, timeout=timeout)
        return ServiceHealthResult(True, 204, "ok", "ready")

    monkeypatch.setattr("apps.daemon.apriald.probe_service_health", probe)
    assert asyncio.run(_loopback_health_check(runtime)) is True
    assert captured == {
        "url": f"{settings.runtime.url}/runtime/health",
        "bearer_token": "daemon-runtime-secret",
        "timeout": 1.0,
    }


def test_service_manager_uses_shared_authenticated_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APRIL_RUNTIME_TOKEN", "manager-runtime-secret")
    captured: dict[str, Any] = {}

    def probe(
        url: str,
        *,
        bearer_token: str | None = None,
        timeout: float,
    ) -> ServiceHealthResult:
        captured.update(url=url, bearer_token=bearer_token, timeout=timeout)
        return ServiceHealthResult(False, 403, "authentication_rejected", "rejected")

    monkeypatch.setattr("apps.runner.service_manager.probe_service_health", probe)
    manager = AprilServiceManager(home=tmp_path)
    runtime_url = f"{manager.settings.runtime.url}/runtime/health"
    assert manager._authenticated_health_getter(runtime_url, 0.5) is False
    assert captured == {
        "url": runtime_url,
        "bearer_token": "manager-runtime-secret",
        "timeout": 0.5,
    }
