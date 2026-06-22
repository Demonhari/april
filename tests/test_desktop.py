from __future__ import annotations

import json
from pathlib import Path

import anyio
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from apps.runner.main import app as runner_app
from apps.runner.service_manager import ServiceInfo, ServiceStatus
from april_common.settings import load_settings
from services.api.server import create_app
from tests.test_core_api import auth, make_container


def test_desktop_static_mount_serves_index(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.get("/desktop/")
    assert response.status_code == 200
    assert "APRIL Desktop" in response.text
    assert response.headers["content-type"].startswith("text/html")


def test_activity_requires_auth(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    assert client.get("/diagnostics/activity").status_code == 403


def test_activity_feed_is_redacted(settings_tmp) -> None:
    audit_path: Path = settings_tmp.audit_path
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    # A worst-case audit line: it carries prompt content, tool arguments with a
    # file path and patch bytes, metadata, and a token. None of these may leak.
    line = {
        "timestamp": "2026-06-22T00:00:00Z",
        "event_type": "tool_executed",
        "request_id": "req-123",
        "approval_id": "appr-456",
        "tool": "patch_applier",
        "agent": "coding_agent",
        "risk_level": "code_write",
        "permission_level": 3,
        "outcome": "consumed",
        "arguments": {"file_path": "/etc/passwd", "patch": "SECRET PATCH BYTES"},
        "metadata": {"artifact_sha256": "deadbeef"},
        "content": "USER PROMPT BODY THAT MUST NOT LEAK",
        "api_token": "tok-should-never-appear",
        "reason": "private reason text",
    }
    audit_path.write_text(json.dumps(line) + "\n", encoding="utf-8")

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.get("/diagnostics/activity?limit=10", headers=auth(settings_tmp))
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    event = data["events"][0]

    # Safe reference fields are present.
    assert event["event_type"] == "tool_executed"
    assert event["request_id"] == "req-123"
    assert event["approval_id"] == "appr-456"
    assert event["tool"] == "patch_applier"
    assert event["risk_level"] == "code_write"

    # Nothing sensitive leaks — neither as keys nor as serialized values.
    banned_keys = {"arguments", "metadata", "content", "api_token", "reason", "patch"}
    assert not (banned_keys & set(event.keys()))
    blob = json.dumps(data)
    secrets = (
        "/etc/passwd",
        "SECRET PATCH BYTES",
        "USER PROMPT BODY",
        "tok-should-never-appear",
        "private reason",
    )
    for secret in secrets:
        assert secret not in blob


def test_activity_limit_is_capped(settings_tmp) -> None:
    audit_path: Path = settings_tmp.audit_path
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"event_type": "ping", "timestamp": f"t{i}", "request_id": str(i)})
        for i in range(500)
    ]
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.get("/diagnostics/activity?limit=99999", headers=auth(settings_tmp))
    assert response.status_code == 200
    assert response.json()["count"] <= 200


class _OkManager:
    """Stub service manager: reports healthy without starting any process."""

    def __init__(self, home: Path) -> None:
        self.home = home
        self.settings = load_settings(root=home)

    def _status(self) -> ServiceStatus:
        ok = ServiceInfo(name="x", pid=1, running=True, healthy=True, log_path=self.home)
        return ServiceStatus(runtime=ok, api=ok)

    def start(self, *, fake_backend: bool = False) -> ServiceStatus:
        return self._status()

    def status(self) -> ServiceStatus:
        return self._status()


def test_desktop_command_resolves_url_without_browser(settings_tmp, monkeypatch) -> None:
    captured: dict[str, str] = {}
    monkeypatch.setattr("apps.runner.main._manager", lambda: _OkManager(settings_tmp.home))
    monkeypatch.setattr(
        "apps.runner.main._open_desktop_browser",
        lambda url: captured.setdefault("url", url) is None or True,
    )
    # If this were reached, pywebview is absent in the test env anyway.
    monkeypatch.setattr("apps.runner.main._open_desktop_native", lambda url, token: False)

    result = CliRunner().invoke(runner_app, ["april", "desktop"])
    assert result.exit_code == 0, result.output
    url = captured["url"]
    assert url.startswith(f"http://{settings_tmp.api.host}:{settings_tmp.api.port}/desktop#token=")
    assert settings_tmp.api.token in url
    # Token is in the fragment, never a query string.
    assert "?token=" not in url


def test_desktop_command_no_open_resolves_without_launch(settings_tmp, monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr("apps.runner.main._manager", lambda: _OkManager(settings_tmp.home))
    monkeypatch.setattr(
        "apps.runner.main._open_desktop_browser",
        lambda url: opened.append(url) or True,
    )
    result = CliRunner().invoke(runner_app, ["april", "desktop", "--no-open"])
    assert result.exit_code == 0, result.output
    assert opened == []
    assert "/desktop" in result.output
