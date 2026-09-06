from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from apps.runner.main import app
from apps.runner.preflight import build_preflight_report
from april_common.audit import AuditVerification
from april_common.credentials import CredentialKey, InMemoryCredentialStore

pytestmark = pytest.mark.usefixtures("clean_april_environment")


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
        tmp_path,
        fake=False,
        port_in_use=_free_port,
        pid_alive=_dead_pid,
        credential_store=InMemoryCredentialStore(),
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


@pytest.mark.parametrize("fake", [False, True])
def test_preflight_blocks_corrupt_audit_even_in_fake_mode(tmp_path: Path, fake: bool) -> None:
    _copy_configs(tmp_path)
    audit_path = tmp_path / "logs" / "audit.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_bytes(b"{}\n{}\n{}\n")
    report = build_preflight_report(
        tmp_path, fake=fake, port_in_use=_free_port, pid_alive=_dead_pid
    )
    check = next(item for item in report.checks if item.name == "audit chain integrity")
    assert report.ok is False
    assert check.status == "fail"
    assert "corrupt" in check.detail
    assert "audit chain integrity" in report.failures


def test_preflight_distinguishes_unavailable_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _copy_configs(tmp_path)

    class UnavailableAudit:
        def verify(self) -> AuditVerification:
            return AuditVerification(
                status="unavailable",
                valid=False,
                corrupt=False,
                anchor_lagged=False,
                record_count=0,
                terminal_sequence=None,
                terminal_hash=None,
            )

    monkeypatch.setattr(
        "apps.runner.preflight.audit_logger_for_settings",
        lambda settings, **kwargs: UnavailableAudit(),
    )
    report = build_preflight_report(
        tmp_path, fake=True, port_in_use=_free_port, pid_alive=_dead_pid
    )
    check = next(item for item in report.checks if item.name == "audit chain integrity")
    assert report.ok is False
    assert check.status == "fail"
    assert "unavailable" in check.detail


@pytest.mark.parametrize("fake", [False, True])
def test_start_preflight_invalid_audit_spawns_no_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: bool
) -> None:
    _copy_configs(tmp_path)
    audit_path = tmp_path / "logs" / "audit.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_bytes(b"{}\n{}\n{}\n")
    actual_preflight = build_preflight_report

    def isolated_preflight(home: Path, **kwargs: object):
        return actual_preflight(home, port_in_use=_free_port, pid_alive=_dead_pid, **kwargs)

    monkeypatch.setattr("apps.runner.main.build_preflight_report", isolated_preflight)
    monkeypatch.setattr("apps.runner.main._manager", lambda: SimpleNamespace(home=tmp_path))
    spawned = False

    def forbidden(_fake: bool):
        nonlocal spawned
        spawned = True
        raise AssertionError("services must not start when audit preflight fails")

    monkeypatch.setattr("apps.runner.main._ensure_services", forbidden)
    args = ["april", "start", "--preflight"]
    if fake:
        args.append("--fake")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1, result.output
    assert spawned is False
    assert "audit chain integrity" in result.output


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


def test_preflight_production_blocks_default_tokens(
    tmp_path: Path, no_real_credential_selection
) -> None:
    _copy_configs(tmp_path)
    _create_gguf_files(tmp_path)
    config = tmp_path / "configs" / "april.yaml"
    data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    data["environment"] = "production"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    report = build_preflight_report(
        tmp_path,
        fake=False,
        port_in_use=_free_port,
        pid_alive=_dead_pid,
        credential_store=InMemoryCredentialStore(
            {
                CredentialKey.API_TOKEN: "local-dev-token",
                CredentialKey.RUNTIME_TOKEN: "local-dev-runtime-token",
            }
        ),
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
