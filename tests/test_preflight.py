from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from apps.runner.preflight import build_preflight_report


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


def _free_port(_host: str, _port: int) -> bool:
    return False


def _dead_pid(_pid: int) -> bool:
    return False


def test_preflight_passes_in_fake_mode(tmp_path: Path) -> None:
    _copy_configs(tmp_path)
    # Force a fake backend so --fake's relaxation is actually exercised.
    config = tmp_path / "configs" / "april.yaml"
    data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    data.setdefault("runtime", {})["backend"] = "fake"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    report = build_preflight_report(
        tmp_path, fake=True, port_in_use=_free_port, pid_alive=_dead_pid
    )
    assert report.ok is True
    assert report.fake is True
    # Fake mode relaxes backend + model-file requirements but nothing else.
    backend = next(c for c in report.checks if c.name == "runtime backend")
    assert backend.status == "warning"
    models = next(c for c in report.checks if c.name == "model files present")
    assert models.status == "pass"


def test_preflight_passes_in_real_mode_when_models_present(tmp_path: Path) -> None:
    _copy_configs(tmp_path)
    _create_gguf_files(tmp_path)
    report = build_preflight_report(
        tmp_path, fake=False, port_in_use=_free_port, pid_alive=_dead_pid
    )
    assert report.ok is True
    models = next(c for c in report.checks if c.name == "model files present")
    assert models.status == "pass"


def test_preflight_fails_real_mode_without_models(tmp_path: Path) -> None:
    _copy_configs(tmp_path)  # no GGUF files created
    report = build_preflight_report(
        tmp_path, fake=False, port_in_use=_free_port, pid_alive=_dead_pid
    )
    assert report.ok is False
    assert "model files present" in report.failures


def test_preflight_fails_real_mode_with_fake_backend(tmp_path: Path) -> None:
    _copy_configs(tmp_path)
    _create_gguf_files(tmp_path)
    config = tmp_path / "configs" / "april.yaml"
    data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    data.setdefault("runtime", {})["backend"] = "fake"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    report = build_preflight_report(
        tmp_path, fake=False, port_in_use=_free_port, pid_alive=_dead_pid
    )
    assert report.ok is False
    assert "runtime backend" in report.failures


def test_preflight_fails_when_foreign_process_holds_port(tmp_path: Path) -> None:
    _copy_configs(tmp_path)
    _create_gguf_files(tmp_path)
    report = build_preflight_report(
        tmp_path,
        fake=False,
        port_in_use=lambda _host, _port: True,  # port occupied
        pid_alive=_dead_pid,  # not our managed process
    )
    assert report.ok is False
    assert any(name.endswith("port available") for name in report.failures)


def test_preflight_production_blocks_default_tokens(tmp_path: Path) -> None:
    _copy_configs(tmp_path)
    _create_gguf_files(tmp_path)
    config = tmp_path / "configs" / "april.yaml"
    data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    data["environment"] = "production"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    report = build_preflight_report(
        tmp_path, fake=False, port_in_use=_free_port, pid_alive=_dead_pid
    )
    # Default tokens in production are rejected at the settings layer, so preflight
    # fails fast (the dedicated token/hardening check is defense in depth).
    assert report.ok is False
    assert report.failures


def test_preflight_stale_pid_is_warning_not_failure(tmp_path: Path) -> None:
    _copy_configs(tmp_path)
    _create_gguf_files(tmp_path)
    run_dir = tmp_path / "data" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "api.pid").write_text("999999", encoding="utf-8")
    report = build_preflight_report(
        tmp_path,
        fake=False,
        port_in_use=_free_port,  # port is free, so the stale pid is just a lock
        pid_alive=_dead_pid,
    )
    stale = next(c for c in report.checks if c.name == "no stale lock files")
    assert stale.status == "warning"
    assert report.ok is True


def test_preflight_redacted(tmp_path: Path) -> None:
    _copy_configs(tmp_path)
    _create_gguf_files(tmp_path)
    report = build_preflight_report(
        tmp_path, fake=False, port_in_use=_free_port, pid_alive=_dead_pid
    )
    import json

    blob = json.dumps(report.model_dump())
    assert str(tmp_path) not in blob
    assert "local-dev-token" not in blob
