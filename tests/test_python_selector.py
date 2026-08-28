from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "scripts" / "select_python.py"


def _fake_python(path: Path, major: int, minor: int) -> None:
    path.write_text(
        f"#!/bin/sh\nif [ \"$1\" = \"-c\" ]; then printf '%s\\n' '{major} {minor}'; fi\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SELECTOR), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_explicit_supported_interpreter_is_reported() -> None:
    result = _run("--path-only", sys.executable)
    assert result.returncode == 0
    assert result.stdout.strip() == sys.executable


def test_explicit_missing_interpreter_is_rejected() -> None:
    result = _run("/does/not/exist")
    assert result.returncode == 1
    assert "does not exist" in result.stderr


def test_explicit_unsupported_interpreter_is_rejected(tmp_path: Path) -> None:
    executable = tmp_path / "python3.14"
    _fake_python(executable, 3, 14)
    result = _run(str(executable))
    assert result.returncode == 1
    assert "requires Python 3.11, 3.12, or 3.13" in result.stderr


def test_fallback_skips_unsupported_and_selects_supported(tmp_path: Path) -> None:
    _fake_python(tmp_path / "python3.13", 3, 14)
    supported = tmp_path / "python3.12"
    _fake_python(supported, 3, 12)
    environment = dict(os.environ)
    environment["PATH"] = str(tmp_path)
    environment.pop("PYTHON_BIN", None)
    environment.pop("PYTHON", None)
    result = _run("--path-only", env=environment)
    assert result.returncode == 0
    assert result.stdout.strip() == str(supported)


def test_fallback_reports_no_supported_interpreter(tmp_path: Path) -> None:
    _fake_python(tmp_path / "python3.14", 3, 14)
    environment = dict(os.environ)
    environment["PATH"] = str(tmp_path)
    environment.pop("PYTHON_BIN", None)
    environment.pop("PYTHON", None)
    result = _run("--path-only", env=environment)
    assert result.returncode == 1
    assert "No supported Python interpreter" in result.stderr


def test_selector_help_is_side_effect_free() -> None:
    result = _run("--help")
    assert result.returncode == 0
    assert "PYTHON_BIN" in result.stdout
