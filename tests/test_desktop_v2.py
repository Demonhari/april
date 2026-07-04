from __future__ import annotations

from pathlib import Path

import anyio
from fastapi.testclient import TestClient

from services.api.server import create_app
from tests.test_core_api import make_container

WEB = Path(__file__).resolve().parents[1] / "apps" / "desktop" / "web"
INDEX = (WEB / "index.html").read_text(encoding="utf-8")
APP_JS = (WEB / "app.js").read_text(encoding="utf-8")


def test_nav_exposes_v2_screens() -> None:
    for screen in ("sessions", "playbooks", "evolution"):
        assert f'data-screen="{screen}"' in INDEX
        assert f"screens.{screen} = async function" in APP_JS


def test_header_shows_wake_status_from_health_booleans() -> None:
    assert 'id="rail-wake"' in INDEX
    assert "state.health.wake" in APP_JS
    # Honest states only: off / muted / on / unknown.
    for word in ('"off"', '"muted"', '"on"', '"unknown"'):
        assert word in APP_JS


def test_chat_feedback_buttons_use_feedback_endpoint() -> None:
    assert "addFeedbackControls" in APP_JS
    assert '"/feedback"' in APP_JS
    assert '{ rating, conversation_id: CONVERSATION_ID }' in APP_JS


def test_sessions_screen_uses_close_endpoint() -> None:
    assert '"/sessions/" + encodeURIComponent(s.id) + "/close"' in APP_JS


def test_evolution_screen_uses_exact_hash_approval() -> None:
    assert '"/evolution/overlays/approve"' in APP_JS
    assert "content_hash: p.content_hash" in APP_JS
    assert '"/evolution/off"' in APP_JS
    assert '"/evolution/on"' in APP_JS
    assert '"/evolution/rollback"' in APP_JS


def test_playbooks_screen_runs_through_api_only() -> None:
    assert '"/playbooks/" + encodeURIComponent(pb.id) + "/run"' in APP_JS
    # The SPA warns that L3+ steps pause for approval instead of auto-running.
    assert "exact-action approval" in APP_JS


def test_no_secrets_or_tokens_in_new_markup() -> None:
    assert "local-dev-token" not in INDEX
    assert "local-dev-token" not in APP_JS


def test_health_reports_wake_booleans_only(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    response = client.get("/health")
    assert response.status_code == 200
    wake = response.json()["wake"]
    assert wake == {"enabled": False, "muted": False}

    settings_tmp.mute_flag_path.parent.mkdir(parents=True, exist_ok=True)
    settings_tmp.mute_flag_path.write_text("muted\n", encoding="utf-8")
    muted = client.get("/health").json()["wake"]
    assert muted["muted"] is True
