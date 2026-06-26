from __future__ import annotations

import json
import os
from pathlib import Path

import anyio
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from apps.runner.main import app as runner_app
from apps.runner.service_manager import ServiceInfo, ServiceStatus
from april_common.settings import load_settings
from services.api.server import create_app
from tests.conftest import FakeRuntimeClient
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


def test_readiness_requires_auth(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    assert client.get("/readiness").status_code == 403


def test_readiness_redacts_tokens_and_paths(settings_tmp) -> None:
    class RuntimeWithPaths(FakeRuntimeClient):
        async def health(self, *, timeout: float | None = None) -> dict[str, object]:
            return {
                "status": "ok",
                "backend": "fake",
                "simulated": True,
                "models": [
                    {
                        "id": "april-brain",
                        "name": "brain",
                        "role": "brain",
                        "backend": "fake",
                        "path": str(settings_tmp.home / "models" / "brain.gguf"),
                        "state": "loaded",
                        "keep_loaded": True,
                        "missing_path": True,
                    }
                ],
            }

        async def models(self) -> dict[str, object]:
            return {
                "models": [
                    {
                        "id": "april-brain",
                        "name": "brain",
                        "role": "brain",
                        "backend": "fake",
                        "path": str(settings_tmp.home / "models" / "brain.gguf"),
                        "state": "loaded",
                        "keep_loaded": True,
                        "missing_path": True,
                    }
                ]
            }

    container = anyio.run(make_container, settings_tmp, RuntimeWithPaths())
    client = TestClient(create_app(container))
    response = client.get("/readiness", headers=auth(settings_tmp))
    assert response.status_code == 200
    data = response.json()
    blob = json.dumps(data)
    assert settings_tmp.api.token not in blob
    assert str(settings_tmp.home) not in blob
    assert str(settings_tmp.database_path) not in blob
    assert data["models"]["registered"][0]["path_basename"] == "brain.gguf"
    assert "/" not in data["models"]["registered"][0]["path_basename"]
    assert data["security"]["api_token"]["status"] == "configured"


def test_latest_verification_report_not_verified(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.get("/verification/report/latest", headers=auth(settings_tmp))
    assert response.status_code == 200
    assert response.json()["status"] == "not_verified"
    assert response.json()["message"] == "not verified yet"


def test_latest_verification_report_redacts_and_ignores_path_query(settings_tmp) -> None:
    report_dir = settings_tmp.home / "data" / "verification"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "report_type": "multi_model",
        "generated_at": "2026-06-26T00:00:00Z",
        "summary": "degraded",
        "real_model_verified": False,
        "verification_level": "partial",
        "real_models_exercised": 1,
        "real_models_passed": 1,
        "core_model_set_verified": False,
        "all_configured_models_verified": False,
        "models": [
            {
                "model_id": "april-brain",
                "role": "brain",
                "backend": "llama_cpp",
                "path": str(settings_tmp.home / "models" / "brain.gguf"),
                "available": False,
                "skipped_reason": f"Missing model file: {settings_tmp.home}/models/brain.gguf",
            }
        ],
        "skipped": [
            {
                "name": "april-brain",
                "reason": f"Missing model file: {settings_tmp.home}/models/brain.gguf",
            }
        ],
        "threshold_failures": [f"april-brain: routing path {settings_tmp.home}/models/brain.gguf"],
        "prompt": "must not leak",
        "generated_text": "must not leak",
        "api_token": settings_tmp.api.token,
    }
    (report_dir / "mac-readiness.json").write_text(json.dumps(report), encoding="utf-8")
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.get(
        "/verification/report/latest?path=/etc/passwd",
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    data = response.json()
    blob = json.dumps(data)
    assert data["status"] == "ok"
    assert data["report"]["models"][0]["path_basename"] == "brain.gguf"
    assert data["report"]["verification_level"] == "partial"
    assert data["report"]["real_models_exercised"] == 1
    assert data["report"]["skipped_count"] == 1
    assert data["report"]["threshold_failure_count"] == 1
    assert str(settings_tmp.home) not in blob
    assert "/etc/passwd" not in blob
    assert settings_tmp.api.token not in blob
    assert "must not leak" not in blob


def _write_voice_live_report(home: Path, basename: str, *, generated_at: str, passed: bool) -> Path:
    report_dir = home / "data" / "verification"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / basename
    report = {
        "report_type": "voice_live",
        "generated_at": generated_at,
        "timestamp": generated_at,
        "platform": "Darwin 25.5.0",
        "summary": "pass" if passed else "degraded",
        "voice_live_verified": passed,
        "recording_success": passed,
        "stt_success": passed,
        "transcript_length": 11,
        "transcription_user_confirmed": passed,
        "tts_success": passed,
        "playback_user_confirmed": passed,
        # Hostile-looking fields that must never reach the sanitized payload.
        "transcript": "SECRET SPOKEN WORDS",
        "input_path": str(home / "data" / "audio_cache" / "voice-live-input.wav"),
        "input_device_name": "MacBook Pro Microphone",
        "output_device_name": "MacBook Pro Speakers",
        "api_token": "secret-token",
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_latest_voice_live_report_is_sanitized(settings_tmp) -> None:
    _write_voice_live_report(
        settings_tmp.home, "voice-live.json", generated_at="2026-06-26T00:00:00Z", passed=True
    )
    container = anyio.run(make_container, settings_tmp)
    with TestClient(create_app(container)) as client:
        response = client.get(
            "/verification/report/latest?type=voice_live", headers=auth(settings_tmp)
        )
    assert response.status_code == 200
    data = response.json()
    report = data["report"]
    assert report["report_type"] == "voice_live"
    assert report["generated_at"] == "2026-06-26T00:00:00Z"
    assert report["summary"] == "pass"
    # The safe voice fields are exposed.
    assert report["voice_live_verified"] is True
    assert report["recording_success"] is True
    assert report["stt_success"] is True
    assert report["tts_success"] is True
    assert report["playback_user_confirmed"] is True
    assert "skipped_count" in report
    # Nothing sensitive leaks: no transcript, audio path, device name, or token.
    blob = json.dumps(data)
    for secret in (
        "SECRET SPOKEN WORDS",
        "voice-live-input.wav",
        "MacBook Pro Microphone",
        "MacBook Pro Speakers",
        "secret-token",
    ):
        assert secret not in blob


def test_failed_voice_live_report_serializes_verified_false(settings_tmp) -> None:
    _write_voice_live_report(
        settings_tmp.home, "voice-live.json", generated_at="2026-06-26T00:00:00Z", passed=False
    )
    container = anyio.run(make_container, settings_tmp)
    with TestClient(create_app(container)) as client:
        response = client.get(
            "/verification/report/latest?type=voice_live", headers=auth(settings_tmp)
        )
    report = response.json()["report"]
    assert report["voice_live_verified"] is False
    assert report["summary"] == "degraded"


def test_latest_real_model_report_is_stable_against_newer_voice_report(settings_tmp) -> None:
    # A real-model (multi_model) report, then a NEWER voice-live report.
    real_model = _write_verification_report(
        settings_tmp.home,
        "mac-readiness.json",
        generated_at="2026-06-25T00:00:00Z",
        verification_level="core",
        summary="pass",
    )
    voice = _write_voice_live_report(
        settings_tmp.home, "voice-live.json", generated_at="2026-06-26T00:00:00Z", passed=True
    )
    os.utime(real_model, (1, 1))
    os.utime(voice, (2, 2))  # voice is strictly newer
    container = anyio.run(make_container, settings_tmp)
    with TestClient(create_app(container)) as client:
        headers = auth(settings_tmp)
        any_latest = client.get("/verification/report/latest", headers=headers).json()
        real_latest = client.get(
            "/verification/report/latest?type=real_model", headers=headers
        ).json()
        voice_latest = client.get(
            "/verification/report/latest?type=voice_live", headers=headers
        ).json()
    # The newest of any kind is the voice report.
    assert any_latest["report"]["report_type"] == "voice_live"
    # But the real-model latest is still the multi_model report, not the voice one.
    assert real_latest["report"]["report_type"] == "multi_model"
    assert real_latest["report"]["verification_level"] == "core"
    assert real_latest["report"]["real_model_verified"] is True
    # And the voice-live latest is the voice report.
    assert voice_latest["report"]["report_type"] == "voice_live"
    assert voice_latest["report"]["voice_live_verified"] is True


def test_latest_voice_live_report_absent_is_not_verified(settings_tmp) -> None:
    # A real-model report exists but no voice report: voice latest is not_verified.
    _write_verification_report(
        settings_tmp.home, "mac-readiness.json", generated_at="2026-06-25T00:00:00Z"
    )
    container = anyio.run(make_container, settings_tmp)
    with TestClient(create_app(container)) as client:
        voice_latest = client.get(
            "/verification/report/latest?type=voice_live", headers=auth(settings_tmp)
        ).json()
    assert voice_latest["status"] == "not_verified"
    assert voice_latest["report"] is None


def _write_verification_report(
    home: Path,
    basename: str,
    *,
    generated_at: str,
    summary: str = "degraded",
    verification_level: str = "partial",
) -> Path:
    report_dir = home / "data" / "verification"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / basename
    report = {
        "report_type": "multi_model",
        "generated_at": generated_at,
        "summary": summary,
        "real_model_verified": verification_level in {"partial", "core", "all"},
        "verification_level": verification_level,
        "real_models_exercised": 1,
        "real_models_passed": 1,
        "any_real_model_exercised": True,
        "any_real_model_passed": True,
        "core_model_set_verified": verification_level in {"core", "all"},
        "all_configured_models_verified": verification_level == "all",
        "models": [
            {
                "model_id": "april-brain",
                "role": "brain",
                "backend": "llama_cpp",
                "path": str(home / "models" / "brain.gguf"),
                "available": True,
            }
        ],
        "skipped": [
            {
                "name": "april-reading",
                "reason": f"Missing model file: {home}/models/reading.gguf",
            }
        ],
        "threshold_failures": [f"april-brain: path {home}/models/brain.gguf"],
        "prompt": "must not leak",
        "generated_text": "must not leak",
        "api_token": "secret-token",
        "raw_tool_args": {"path": "/etc/passwd"},
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_verification_report_history_requires_auth(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    assert client.get("/verification/reports").status_code == 403
    assert client.get("/verification/reports/mac-readiness.json").status_code == 403


def test_verification_report_history_no_reports(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    with TestClient(create_app(container)) as client:
        response = client.get("/verification/reports", headers=auth(settings_tmp))
    assert response.status_code == 200
    assert response.json() == {
        "status": "not_verified",
        "message": "not verified yet",
        "reports": [],
        "count": 0,
    }


def test_verification_report_history_sorted_and_sanitized(settings_tmp) -> None:
    older = _write_verification_report(
        settings_tmp.home,
        "older.json",
        generated_at="2026-06-25T00:00:00Z",
        verification_level="partial",
    )
    newer = _write_verification_report(
        settings_tmp.home,
        "newer.json",
        generated_at="2026-06-26T00:00:00Z",
        verification_level="all",
        summary="pass",
    )
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    container = anyio.run(make_container, settings_tmp)
    with TestClient(create_app(container)) as client:
        response = client.get("/verification/reports", headers=auth(settings_tmp))
        detail = client.get("/verification/reports/newer.json", headers=auth(settings_tmp))
    assert response.status_code == 200
    data = response.json()
    assert [item["basename"] for item in data["reports"]] == ["newer.json", "older.json"]
    latest = data["reports"][0]
    assert latest["verification_level"] == "all"
    assert latest["skipped_count"] == 1
    assert latest["threshold_failure_count"] == 1
    blob = json.dumps(data)
    assert str(settings_tmp.home) not in blob
    assert "must not leak" not in blob
    assert "secret-token" not in blob
    assert "/etc/passwd" not in blob
    assert detail.status_code == 200
    assert detail.json()["report"]["basename"] == "newer.json"


def test_verification_report_history_rejects_traversal_and_query_path(settings_tmp) -> None:
    _write_verification_report(
        settings_tmp.home,
        "mac-readiness.json",
        generated_at="2026-06-26T00:00:00Z",
    )
    container = anyio.run(make_container, settings_tmp)
    with TestClient(create_app(container)) as client:
        query = client.get("/verification/reports?path=/etc/passwd", headers=auth(settings_tmp))
        traversal = client.get(
            "/verification/reports/../mac-readiness.json",
            headers=auth(settings_tmp),
        )
        encoded = client.get(
            "/verification/reports/%2Fetc%2Fpasswd",
            headers=auth(settings_tmp),
        )
        assert (
            client.get(
                "/verification/reports/mac-readiness.json?path=/etc/passwd",
                headers=auth(settings_tmp),
            ).status_code
            == 400
        )
    assert query.status_code == 400
    assert traversal.status_code in (400, 404)
    assert encoded.status_code in (400, 404)


def test_verification_report_history_rejects_symlink_escape(settings_tmp, tmp_path: Path) -> None:
    report_dir = settings_tmp.home / "data" / "verification"
    report_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"report_type": "multi_model"}), encoding="utf-8")
    symlink = report_dir / "escape.json"
    symlink.symlink_to(outside)
    container = anyio.run(make_container, settings_tmp)
    with TestClient(create_app(container)) as client:
        listed = client.get("/verification/reports", headers=auth(settings_tmp)).json()
        detail = client.get("/verification/reports/escape.json", headers=auth(settings_tmp))
    assert listed["reports"] == []
    assert detail.status_code == 400


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
