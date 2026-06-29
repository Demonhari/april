"""Clean dev install (no optional ``[runtime]`` extra) readiness semantics.

The base/dev install intentionally omits ``llama-cpp-python``. These tests pin
``importlib.util.find_spec('llama_cpp')`` to *absent* (via the ``llama_cpp_missing``
fixture) so they assert the honest production behaviour regardless of whether the
host machine happens to have the extra installed:

* offline readiness reports the runtime extra as missing,
* the daily-driver core real-model rollup is a hard blocker,
* the onboarding checklist's first blocker is install-dependencies.

Real-model readiness must never be claimed when the runtime extra is absent.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from apps.runner.daily_driver import build_daily_driver_report
from apps.runner.readiness import build_readiness_report
from apps.runner.setup_checklist import build_setup_checklist


def _copy_configs(home: Path) -> None:
    shutil.copytree(Path.cwd() / "configs", home / "configs")


def _create_gguf_files(home: Path) -> None:
    models = yaml.safe_load((home / "configs" / "models.yaml").read_text(encoding="utf-8"))
    for model in models["models"].values():
        if model.get("backend") != "llama_cpp":
            continue
        path = home / model["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"GGUF stub")


def _ready_home(tmp_path: Path) -> Path:
    # Even with every GGUF present, a missing runtime extra must still block.
    home = tmp_path / "home"
    home.mkdir()
    _copy_configs(home)
    _create_gguf_files(home)
    return home


def test_readiness_reports_runtime_extra_missing(tmp_path: Path, llama_cpp_missing: None) -> None:
    home = _ready_home(tmp_path)
    readiness = build_readiness_report(home)
    assert readiness.llama_cpp_python_available is False
    check = next(c for c in readiness.checks if c.name == "llama-cpp-python")
    assert check.status == "blocker"
    assert check.action == "pip install -e '.[runtime]'"
    assert "llama-cpp-python" in readiness.blockers
    # Prerequisites are not met, so the offline preflight may not claim readiness.
    assert readiness.real_model_preflight_ready is False
    assert readiness.real_model_ready is False


def test_daily_driver_reports_real_model_blocker(tmp_path: Path, llama_cpp_missing: None) -> None:
    home = _ready_home(tmp_path)
    report = build_daily_driver_report(home)
    assert report.core_real_model == "blocker"
    check = next(c for c in report.checks if c.name == "llama-cpp-python")
    assert check.status == "blocker"
    assert check.next_command == "pip install -e '.[runtime]'"


def test_checklist_first_blocker_is_install_dependencies(
    tmp_path: Path, llama_cpp_missing: None
) -> None:
    home = _ready_home(tmp_path)
    checklist = build_setup_checklist(home)
    install = checklist.steps[0]
    assert install.title == "install dependencies"
    assert install.status == "blocker"
    # The headline next command points at install-dependencies (the first blocker),
    # even though the GGUF files are all present.
    assert checklist.next_command == install.command
    assert checklist.next_command == "pip install -e '.[runtime]'"
