"""Audited hard mute/unmute: API route, audit trail, CLI behaviour."""

from __future__ import annotations

import json
from typing import Any

import anyio
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from apps.cli.client import ApiOfflineError, ApiResponseError
from apps.cli.main import app as cli_app
from services.api.server import create_app
from tests.test_core_api import auth, make_container


def test_mute_route_flips_flag_and_audits(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)

    status = client.get("/wake/mute", headers=headers)
    assert status.status_code == 200
    assert status.json() == {"muted": False, "state": "idle"}

    muted = client.post("/wake/mute", json={"muted": True}, headers=headers)
    assert muted.status_code == 200
    assert muted.json() == {"muted": True, "state": "muted", "audited": True}
    assert settings_tmp.mute_flag_path.exists()

    unmuted = client.post("/wake/mute", json={"muted": False}, headers=headers)
    assert unmuted.status_code == 200
    assert unmuted.json() == {"muted": False, "state": "idle", "audited": True}
    assert not settings_tmp.mute_flag_path.exists()

    audit_events = [
        json.loads(line)
        for line in settings_tmp.audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert sum(event["event_type"] == "wake_mute_changed" for event in audit_events) == 2
    assert {event["muted"] for event in audit_events} == {True, False}
    assert all(event["actor"] == "local-user" for event in audit_events)
    # The mute flag path must never leak through the API or audit surface.
    assert str(settings_tmp.mute_flag_path) not in json.dumps(audit_events)


def test_mute_route_requires_authentication(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))

    assert client.get("/wake/mute").status_code == 403
    assert client.post("/wake/mute", json={"muted": True}).status_code == 403
    assert not settings_tmp.mute_flag_path.exists()

    wrong = client.post(
        "/wake/mute",
        json={"muted": True},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert wrong.status_code == 403
    assert not settings_tmp.mute_flag_path.exists()


def test_health_reflects_mute_state(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    headers = auth(settings_tmp)

    assert client.get("/diagnostics", headers=headers).json()["wake"]["muted"] is False
    client.post("/wake/mute", json={"muted": True}, headers=headers)
    assert client.get("/diagnostics", headers=headers).json()["wake"]["muted"] is True


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def post(self, path: str, payload: dict[str, Any], *, auth: bool = True) -> Any:
        self.calls.append((path, payload))
        return {"muted": payload["muted"], "audited": True}


def test_cli_mute_uses_api_when_available(settings_tmp, monkeypatch) -> None:
    fake = _RecordingClient()
    monkeypatch.setattr("apps.cli.main.client", lambda: fake)
    runner = CliRunner()

    result = runner.invoke(cli_app, ["mute"])
    assert result.exit_code == 0, result.output
    assert ("/wake/mute", {"muted": True}) in fake.calls
    # The audited API path was used; the CLI never touched the flag directly.
    assert not settings_tmp.mute_flag_path.exists()
    assert "unaudited" not in result.output

    result_off = runner.invoke(cli_app, ["mute", "--off"])
    assert result_off.exit_code == 0, result_off.output
    assert ("/wake/mute", {"muted": False}) in fake.calls


class _OfflineClient:
    async def post(self, path: str, payload: dict[str, Any], *, auth: bool = True) -> Any:
        raise ApiOfflineError("offline")


def test_cli_mute_offline_fallback_is_explicitly_unaudited(settings_tmp, monkeypatch) -> None:
    monkeypatch.setattr("apps.cli.main.client", lambda: _OfflineClient())
    runner = CliRunner()

    result = runner.invoke(cli_app, ["mute"])
    assert result.exit_code == 0, result.output
    assert settings_tmp.mute_flag_path.exists()
    assert "unaudited_fallback" in result.output

    result_off = runner.invoke(cli_app, ["mute", "--off"])
    assert result_off.exit_code == 0, result_off.output
    assert not settings_tmp.mute_flag_path.exists()


class _DeniedClient:
    async def post(self, path: str, payload: dict[str, Any], *, auth: bool = True) -> Any:
        raise ApiResponseError("Invalid bearer token.")


def test_cli_mute_never_falls_back_on_api_denial(settings_tmp, monkeypatch) -> None:
    # A reachable API that refuses the request (bad token) must not be
    # bypassed by an unaudited local write.
    monkeypatch.setattr("apps.cli.main.client", lambda: _DeniedClient())
    result = CliRunner().invoke(cli_app, ["mute"])
    assert result.exit_code == 1
    assert not settings_tmp.mute_flag_path.exists()
