from __future__ import annotations

import json
import re
import shutil
import subprocess

import anyio
import pytest
from fastapi.testclient import TestClient

from april_common.settings import project_root
from services.api.server import create_app
from tests.test_core_api import auth, make_container

WEB = project_root() / "apps" / "desktop" / "web"


def _read(name: str) -> str:
    return (WEB / name).read_text(encoding="utf-8")


# --- primary dashboard screen ---------------------------------------------
def test_dashboard_is_default_screen() -> None:
    html = _read("index.html")
    app = _read("app.js")
    # Dashboard is the first nav entry and the SPA's default screen constant.
    assert 'data-screen="dashboard"' in html
    assert ">Dashboard<" in html
    assert 'const DEFAULT_SCREEN = "dashboard";' in app
    # Boot navigates to the default (dashboard) screen, not the plain Chat page.
    assert "navigate(DEFAULT_SCREEN)" in app
    # Dashboard is the FIRST nav entry (primary, not secondary). Parsed from the
    # ordered nav buttons semantically rather than by raw source offset, so the
    # test asserts the real property and survives reformatting.
    nav_order = re.findall(r'class="navitem" data-screen="([a-z]+)"', html)
    assert nav_order, "no nav buttons found"
    assert nav_order[0] == "dashboard"
    assert "chat" in nav_order
    assert "readiness" in nav_order
    assert ">Readiness<" in html


def test_dashboard_helpers_loaded_before_app() -> None:
    html = _read("index.html")
    # Pure helpers must be defined before the SPA that consumes them, and after
    # the token bridge that the SPA's boot order depends on.
    assert html.index("token_bridge.js") < html.index("dashboard_helpers.js")
    assert html.index("dashboard_helpers.js") < html.index("app.js")


# --- no external assets / build step --------------------------------------
def test_no_external_assets_or_cdns() -> None:
    forbidden = (
        "http://",
        "https://",
        "//cdn",
        "cdn.",
        "googleapis",
        "gstatic",
        "unpkg",
        "jsdelivr",
        "cdnjs",
        "@import",
        "<img",
        ".woff",
        ".ttf",
        "integrity=",
        "crossorigin",
    )
    # Every served web asset, including the security-critical token bridge.
    for name in ("index.html", "styles.css", "app.js", "dashboard_helpers.js", "token_bridge.js"):
        text = _read(name).lower()
        for needle in forbidden:
            assert needle not in text, f"{name} references external asset pattern {needle!r}"


# --- token safety ----------------------------------------------------------
def test_token_not_embedded_in_html() -> None:
    html = _read("index.html")
    # The only legitimate appearance of "token" is the bridge script filename.
    assert "token" not in html.lower().replace("token_bridge.js", "")


def test_app_never_persists_or_logs_token() -> None:
    for name in ("app.js", "dashboard_helpers.js"):
        text = _read(name)
        assert "localStorage." not in text
        assert "sessionStorage." not in text
        assert "document.cookie" not in text
        assert ".setItem(" not in text
        for line in text.splitlines():
            if "console" in line:
                assert "TOKEN" not in line


def test_readiness_screen_is_static_and_token_free() -> None:
    app = _read("app.js")
    assert "screens.readiness" in app
    segment = app[app.index("screens.readiness") : app.index("screens.status")]
    assert "/readiness" in segment
    assert "/verification/report/latest" in segment
    assert "/verification/reports" in segment
    assert "local-dev-token" not in segment
    assert "TOKEN" not in segment
    assert "localStorage" not in segment
    assert "sessionStorage" not in segment
    assert "document.cookie" not in segment
    assert "run april verify --all-configured-models --require-real-model" in segment
    assert "run april voice verify-live" in segment
    assert "run april setup models" in segment
    assert "run april setup voice" in segment
    assert "run april setup app-stub" in segment
    assert "bindCommandCopies();" in segment
    assert "navigator.clipboard.writeText" in app


def test_authenticated_polling_starts_after_token() -> None:
    app = _read("app.js")
    acquire_index = app.index("acquireToken(window)")
    guard_index = app.index("if (!TOKEN)")
    # The dashboard fetches on mount, so polling + navigation must be gated on a
    # successful token acquisition. Pin the actual CALL site ("startPolling();"
    # — the ";" distinguishes the call from the "function startPolling()"
    # definition) strictly after the guard, so a relocated definition cannot
    # satisfy the check while the call still runs early.
    assert app.count("startPolling();") == 1
    call_index = app.index("startPolling();")
    assert acquire_index < guard_index < call_index
    assert app.index("navigate(DEFAULT_SCREEN)") > guard_index
    # The token guard short-circuits boot (return) before any authenticated work.
    assert "return;" in app[guard_index:call_index]
    # The silent poller is the only thing that issues background fetches, and it
    # is only wired up inside startPolling().
    assert "setInterval(refreshConnection" in app


# --- approvals are exact-ID only ------------------------------------------
def test_approvals_use_exact_id_endpoints() -> None:
    app = _read("app.js")
    # Pin the full call shape: POST to the exact endpoint with ONLY the exact
    # approval id in the body. A GET/wrong-method or wrong-id regression fails.
    assert re.search(r'api\(\s*"POST",\s*"/tools/approve",\s*\{\s*approval_id:\s*ap\.id', app)
    assert re.search(r'api\(\s*"POST",\s*"/tools/deny",\s*\{\s*approval_id:\s*ap\.id', app)
    # A chat "yes" must never approve: the chat path has no approve call.
    chat_segment = app[
        app.index("async function streamChat") : app.index("function flashApprovals")
    ]
    assert "/tools/approve" not in chat_segment


# --- explicit status wording (real model verified / voice / approval) ------
def test_status_wording_helpers_and_render_present() -> None:
    helpers = _read("dashboard_helpers.js")
    app = _read("app.js")
    # The pure helpers expose the explicit Status-screen wording.
    assert "realModelVerifiedLabel" in helpers
    assert "real model verified: " in helpers
    assert "voiceLiveWarning" in helpers
    assert "not live-verified" in helpers
    assert "APPROVAL_DISCLAIMER" in helpers
    assert "not approval" in helpers
    assert "run april approve" in helpers
    # The verification card renders the explicit "real model verified" field and a
    # voice-not-live-verified warning; the approvals screen shows the disclaimer.
    # Real-model status comes from the real-model report and voice status from the
    # voice report, so a newer report of one kind cannot overwrite the other.
    assert "D.realModelVerifiedLabel(realModel)" in app
    assert "D.voiceLiveWarning(voice)" in app
    assert "/verification/report/latest?type=real_model" in app
    assert "/verification/report/latest?type=voice_live" in app
    assert "renderLatestReport(latest, latestRealModel, latestVoice)" in app
    assert "D.APPROVAL_DISCLAIMER" in app


# --- activity feed redaction ----------------------------------------------
def test_activity_rendering_uses_redacted_projection() -> None:
    app = _read("app.js")
    # Both feed renderers route events through the allowlist projection helper.
    assert app.count("D.activityRow(e)") >= 2
    # No raw audit field that could carry prompt/arg/secret data is read off an
    # event object directly.
    for leak in ("e.arguments", "e.content", "e.patch", ".api_token", "e.metadata", "e.reason"):
        assert leak not in app, f"activity rendering reads sensitive field {leak!r}"


# --- runtime simulated badge ----------------------------------------------
def test_simulated_runtime_badge_present() -> None:
    app = _read("app.js")
    assert "SIMULATED" in app
    assert "simulated" in app
    assert "backend.simulated" in app


# --- model load/unload via existing endpoints -----------------------------
def test_model_load_unload_endpoints() -> None:
    app = _read("app.js")
    # Pin method + body: POST the exact endpoint with the model id, so a
    # GET/wrong-method or missing-model_id regression is caught.
    assert re.search(r'api\(\s*"POST",\s*"/runtime/models/load",\s*\{\s*model_id:\s*modelId', app)
    assert re.search(r'api\(\s*"POST",\s*"/runtime/models/unload",\s*\{\s*model_id:\s*modelId', app)


# --- wide cockpit layout --------------------------------------------------
def test_wide_cockpit_layout_present() -> None:
    css = _read("styles.css")
    app = _read("app.js")
    assert ".cockpit" in css
    assert "grid-template-areas" in css
    # Responsive breakpoints: wide three-column and medium two-column.
    assert "@media (min-width: 1180px)" in css
    assert "@media (min-width: 800px)" in css
    assert "prefers-reduced-motion" in css
    # Design system variables.
    for token in (
        "--april-bg",
        "--april-panel",
        "--april-line",
        "--april-cyan",
        "--april-green",
        "--april-orange",
        "--april-red",
        "--april-muted",
        "--april-text",
    ):
        assert token in css, f"missing design token {token}"
    # The dashboard screen builds the cockpit grid and the router orbit.
    assert 'el("div", "cockpit")' in app
    assert "renderOrbit" in app


# --- node behavioural test wrapper ----------------------------------------
def test_desktop_dashboard_helpers_under_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available to run the JS behaviour tests")
    script = project_root() / "tests" / "js" / "desktop_dashboard.test.cjs"
    result = subprocess.run(
        [node, str(script)], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


# --- live API contract the dashboard polls --------------------------------
def test_dashboard_polled_endpoints_are_authenticated_and_shaped(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    # Context manager runs the FastAPI lifespan so the container is closed
    # cleanly on exit (no leaked SQLite/TestClient resources).
    with TestClient(create_app(container)) as client:
        headers = auth(settings_tmp)
        # Every GET endpoint the cockpit polls must require auth and return its
        # documented envelope, so the data-driven UI never silently degrades.
        health = client.get("/health")
        assert health.status_code == 200  # health is unauthenticated
        # The rail/telemetry read /health, so its path redaction must hold: the
        # real database path must never leak through this surface.
        health_body = health.json()
        assert health_body["database"]["path"] == "[REDACTED]"
        assert str(settings_tmp.database_path) not in json.dumps(health_body)
        for path in (
            "/approvals",
            "/runtime/models",
            "/reminders",
            "/tasks",
            "/readiness",
            "/verification/report/latest",
            "/verification/reports",
        ):
            assert client.get(path).status_code in (401, 403), path
            body = client.get(path, headers=headers).json()
            assert isinstance(body, dict), path
        assert "approvals" in client.get("/approvals", headers=headers).json()
        assert "models" in client.get("/runtime/models", headers=headers).json()
        assert "reminders" in client.get("/reminders", headers=headers).json()
        assert "tasks" in client.get("/tasks", headers=headers).json()
        readiness = client.get("/readiness", headers=headers).json()
        assert "models" in readiness
        assert str(settings_tmp.home) not in json.dumps(readiness)
        latest = client.get("/verification/report/latest", headers=headers).json()
        assert latest["message"] == "not verified yet"
        reports = client.get("/verification/reports", headers=headers).json()
        assert reports["message"] == "not verified yet"
        assert reports["reports"] == []
        activity = client.get("/diagnostics/activity?limit=80", headers=headers).json()
        assert "events" in activity
        assert "count" in activity
        briefing = client.get("/scheduler/briefing/preview", headers=headers).json()
        assert "title" in briefing
        assert "body" in briefing
