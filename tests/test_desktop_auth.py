from __future__ import annotations

import shutil
import subprocess

import pytest

from apps.runner.main import DesktopTokenBridge
from april_common.settings import project_root

WEB = project_root() / "apps" / "desktop" / "web"


def _read(name: str) -> str:
    return (WEB / name).read_text(encoding="utf-8")


def test_desktop_token_bridge_behaviour_under_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available to run the JS behaviour tests")
    script = project_root() / "tests" / "js" / "desktop_token_bridge.test.cjs"
    result = subprocess.run(
        [node, str(script)], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_spa_uses_async_bridge_not_sync_global() -> None:
    app = _read("app.js")
    bridge = _read("token_bridge.js")
    # No synchronously-injected global is assumed anywhere any more.
    assert "__APRIL_TOKEN__" not in app
    assert "__APRIL_TOKEN__" not in bridge
    # Native path uses the async bridge and waits for readiness.
    assert "win.pywebview.api.get_token" in bridge
    assert "pywebviewready" in bridge
    assert "await window.AprilDesktopAuth.acquireToken(window)" in app


def test_spa_defers_requests_until_token_acquired() -> None:
    app = _read("app.js")
    acquire_index = app.index("acquireToken(window)")
    # The authenticated polling loop must only run after token acquisition.
    assert app.rindex("refreshConnection()") > acquire_index
    assert "if (!TOKEN)" in app
    assert "return;" in app


def test_spa_never_persists_or_logs_token() -> None:
    for name in ("app.js", "token_bridge.js"):
        text = _read(name)
        # Check for actual storage access (dotted use), not comment mentions.
        assert "localStorage." not in text
        assert "sessionStorage." not in text
        assert "document.cookie" not in text
        assert ".setItem(" not in text
        for line in text.splitlines():
            # The in-memory secret variable is TOKEN; it must never be logged.
            if "console" in line:
                assert "TOKEN" not in line


def test_index_loads_bridge_before_app_and_omits_token() -> None:
    html = _read("index.html")
    assert html.index("token_bridge.js") < html.index("app.js")
    # The token is never embedded in the served HTML.
    assert "token" not in html.lower().replace("token_bridge.js", "")


def test_native_token_bridge_minimal_surface() -> None:
    bridge = DesktopTokenBridge("secret-xyz")
    assert bridge.get_token() == "secret-xyz"
    public = [name for name in dir(bridge) if not name.startswith("_")]
    assert public == ["get_token"]
