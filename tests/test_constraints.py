from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
    assert match is not None
    return match.group(1).lower().replace("_", "-")


def _pinned_constraints() -> set[str]:
    pinned: set[str] = set()
    for line in (ROOT / "constraints-dev.txt").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "==" in stripped:
            pinned.add(_requirement_name(stripped))
    return pinned


def _documented_build_exceptions() -> set[str]:
    documented: set[str] = set()
    marker = "# build-system-unpinned:"
    for line in (ROOT / "constraints-dev.txt").read_text(encoding="utf-8").splitlines():
        if marker in line:
            documented.add(_requirement_name(line.split(marker, 1)[1].strip()))
    return documented


def test_build_system_requirements_are_pinned_or_documented() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    build_requires = {
        _requirement_name(requirement)
        for requirement in pyproject.get("build-system", {}).get("requires", [])
    }
    covered = _pinned_constraints() | _documented_build_exceptions()
    assert build_requires <= covered


def test_base_constraints_do_not_include_optional_model_or_voice_artifacts() -> None:
    entries = "\n".join(
        line.strip().lower()
        for line in (ROOT / "constraints-dev.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    forbidden = (
        "llama-cpp-python",
        "sounddevice",
        "openwakeword",
        "whisper.cpp",
        "piper",
        ".gguf",
        ".onnx",
    )
    for needle in forbidden:
        assert needle not in entries


def test_setup_mac_uses_constraints_for_editable_installs() -> None:
    script = (ROOT / "scripts" / "setup_mac.sh").read_text(encoding="utf-8")
    assert '.venv/bin/pip install -e ".[${INSTALL_EXTRAS}]" -c constraints-dev.txt' in script
    assert ".venv/bin/pip install -e . -c constraints-dev.txt" in script


def test_install_run_april_uses_constraints_for_dev_editable_install() -> None:
    script = (ROOT / "scripts" / "install_run_april.sh").read_text(encoding="utf-8")
    assert '.venv/bin/pip install -e ".[dev]" -c constraints-dev.txt' in script


def test_setup_mac_help_documents_install_modes() -> None:
    script = (ROOT / "scripts" / "setup_mac.sh").read_text(encoding="utf-8")
    help_block = script.split("HELP", 2)[1]
    assert "--runtime" in help_block
    assert "--voice" in help_block
    assert "--base" in help_block


def test_generated_verification_and_app_stub_outputs_are_ignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    required = {
        "data/verification/",
        "dist/",
        "*.app/",
        ".april_tmp/",
        "__pycache__/",
        "*.py[cod]",
        "models/*.gguf",
        "models/*.bin",
        "data/audio_cache/",
    }
    assert required <= set(gitignore)
