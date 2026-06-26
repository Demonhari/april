from __future__ import annotations

import os
import subprocess
from pathlib import Path

from april_common.settings import project_root


def test_macos_app_stub_contains_no_tokens_or_models(tmp_path: Path) -> None:
    output = tmp_path / "APRIL.app"
    env = dict(os.environ)
    env["APRIL_APP_STUB_OUTPUT"] = str(output)
    result = subprocess.run(
        ["bash", str(project_root() / "scripts" / "create_macos_app_stub.sh")],
        cwd=project_root(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    launcher = output / "Contents" / "MacOS" / "APRIL"
    info = output / "Contents" / "Info.plist"
    assert launcher.exists()
    assert info.exists()
    combined = launcher.read_text(encoding="utf-8") + info.read_text(encoding="utf-8")
    forbidden = (
        "local-dev-token",
        "local-dev-runtime-token",
        ".gguf",
        "models/",
        "sudo",
        "codesign",
        "notarytool",
        "launchctl",
    )
    for needle in forbidden:
        assert needle not in combined
    assert "run april desktop" in combined
    assert "Unsigned local development launcher" in combined
