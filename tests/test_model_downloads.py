from __future__ import annotations

import hashlib
import json
import shutil
import types
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from apps.runner.main import app
from apps.runner.model_downloads import (
    ModelDownloadReport,
    load_model_download_manifest,
    run_model_downloads,
    validate_gguf_file,
    write_model_download_report,
)
from april_common.errors import ConfigError

GGUF_PAYLOAD = b"GGUF" + (b"\x00" * 64)


def _copy_configs(home: Path) -> None:
    shutil.copytree(Path.cwd() / "configs", home / "configs")


def _write_payload(_url: str, part_path: Path, _token: str | None) -> None:
    part_path.write_bytes(GGUF_PAYLOAD)


def test_manifest_loads_and_validates() -> None:
    manifest = load_model_download_manifest(Path.cwd())

    assert set(manifest) >= {"brain", "coding", "reading"}
    assert manifest["brain"].repo_id == "ibm-granite/granite-3.3-2b-instruct-GGUF"
    assert manifest["coding"].required_for_full_activation is True
    assert manifest["reading"].target_path.as_posix() == "models/qwen3-0.6b-q8_0.gguf"


def test_dry_run_downloads_nothing(tmp_path: Path) -> None:
    _copy_configs(tmp_path)

    def _fail(_url: str, _part_path: Path, _token: str | None) -> None:
        raise AssertionError("dry run must not call downloader")

    report = run_model_downloads(
        tmp_path,
        all_core=True,
        download_func=_fail,
    )

    assert report.applied is False
    assert [entry.status for entry in report.entries] == [
        "would_download",
        "would_download",
        "would_download",
    ]
    assert not (tmp_path / "models").exists()


def test_apply_yes_all_core_downloads_manifest_roles(tmp_path: Path) -> None:
    _copy_configs(tmp_path)
    calls: list[str] = []

    def _record(url: str, part_path: Path, token: str | None) -> None:
        calls.append(url)
        assert token is None
        part_path.write_bytes(GGUF_PAYLOAD)

    report = run_model_downloads(
        tmp_path,
        all_core=True,
        apply=True,
        yes=True,
        download_func=_record,
    )

    assert report.applied is True
    assert report.registration_applied is True
    assert report.selected_roles == ["brain", "coding", "reading"]
    assert len(calls) == 3
    assert (tmp_path / "models" / "granite3.3-2b-q4_k_m.gguf").exists()
    assert (tmp_path / "models" / "qwen3-1.7b-q8_0.gguf").exists()
    assert (tmp_path / "models" / "qwen3-0.6b-q8_0.gguf").exists()


def test_role_specific_download_only_downloads_that_role(tmp_path: Path) -> None:
    _copy_configs(tmp_path)
    calls: list[str] = []

    def _record(url: str, part_path: Path, _token: str | None) -> None:
        calls.append(url)
        part_path.write_bytes(GGUF_PAYLOAD)

    report = run_model_downloads(
        tmp_path,
        role="reading",
        apply=True,
        yes=True,
        download_func=_record,
    )

    assert report.selected_roles == ["reading"]
    assert len(calls) == 1
    assert "Qwen3-0.6B-GGUF" in calls[0]
    assert not (tmp_path / "models" / "granite3.3-2b-q4_k_m.gguf").exists()
    assert (tmp_path / "models" / "qwen3-0.6b-q8_0.gguf").exists()


def test_existing_file_is_skipped_with_skip_existing(tmp_path: Path) -> None:
    _copy_configs(tmp_path)
    target = tmp_path / "models" / "granite3.3-2b-q4_k_m.gguf"
    target.parent.mkdir()
    target.write_bytes(GGUF_PAYLOAD)

    def _fail(_url: str, _part_path: Path, _token: str | None) -> None:
        raise AssertionError("existing target should be skipped")

    report = run_model_downloads(
        tmp_path,
        role="brain",
        apply=True,
        yes=True,
        skip_existing=True,
        download_func=_fail,
    )

    assert report.entries[0].status == "skipped_existing"
    assert target.read_bytes() == GGUF_PAYLOAD


def test_existing_file_fails_without_force_or_skip_existing(tmp_path: Path) -> None:
    _copy_configs(tmp_path)
    target = tmp_path / "models" / "granite3.3-2b-q4_k_m.gguf"
    target.parent.mkdir()
    target.write_bytes(GGUF_PAYLOAD)

    with pytest.raises(ConfigError, match="already exists"):
        run_model_downloads(
            tmp_path,
            role="brain",
            apply=True,
            yes=True,
            download_func=_write_payload,
        )

    assert target.read_bytes() == GGUF_PAYLOAD


def test_failed_download_removes_only_current_part(tmp_path: Path) -> None:
    _copy_configs(tmp_path)
    target = tmp_path / "models" / "granite3.3-2b-q4_k_m.gguf"
    unrelated = tmp_path / "models" / "unrelated.gguf"
    unrelated.parent.mkdir()
    unrelated.write_bytes(GGUF_PAYLOAD)

    def _fail(_url: str, part_path: Path, _token: str | None) -> None:
        part_path.write_bytes(b"partial")
        raise OSError("network failed")

    with pytest.raises(OSError, match="network failed"):
        run_model_downloads(
            tmp_path,
            role="brain",
            apply=True,
            yes=True,
            download_func=_fail,
        )

    assert not target.exists()
    assert not target.with_name(f"{target.name}.part").exists()
    assert unrelated.read_bytes() == GGUF_PAYLOAD


def test_failed_validation_removes_incomplete_target(tmp_path: Path) -> None:
    _copy_configs(tmp_path)
    target = tmp_path / "models" / "granite3.3-2b-q4_k_m.gguf"

    def _bad_payload(_url: str, part_path: Path, _token: str | None) -> None:
        part_path.write_bytes(b"not gguf")

    with pytest.raises(ConfigError, match=r"too small|GGUF magic"):
        run_model_downloads(
            tmp_path,
            role="brain",
            apply=True,
            yes=True,
            download_func=_bad_payload,
        )

    assert not target.exists()
    assert not target.with_name(f"{target.name}.part").exists()


def test_successful_download_atomically_creates_target_and_records_sha(tmp_path: Path) -> None:
    _copy_configs(tmp_path)
    target = tmp_path / "models" / "granite3.3-2b-q4_k_m.gguf"

    report = run_model_downloads(
        tmp_path,
        role="brain",
        apply=True,
        yes=True,
        download_func=_write_payload,
    )

    assert target.exists()
    assert not target.with_name(f"{target.name}.part").exists()
    expected = hashlib.sha256(GGUF_PAYLOAD).hexdigest()
    assert report.entries[0].sha256 == expected
    assert report.entries[0].size_bytes == len(GGUF_PAYLOAD)


def test_fake_gguf_header_validation_passes(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(GGUF_PAYLOAD)

    validate_gguf_file(path)


def test_non_gguf_payload_fails_validation(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"NOPE" + (b"\x00" * 64))

    with pytest.raises(ConfigError, match="GGUF magic"):
        validate_gguf_file(path)


def test_report_redacts_absolute_paths_and_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_configs(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_token_value")

    report = run_model_downloads(
        tmp_path,
        role="brain",
        apply=True,
        yes=True,
        download_func=_write_payload,
    )
    out = tmp_path / "report.json"
    write_model_download_report(report, out)
    text = out.read_text(encoding="utf-8")

    assert str(tmp_path) not in text
    assert "hf_secret_token_value" not in text
    payload = json.loads(text)
    assert payload["entries"][0]["target_path"] == "models/granite3.3-2b-q4_k_m.gguf"


def test_download_report_does_not_mark_real_model_verification_passed(tmp_path: Path) -> None:
    _copy_configs(tmp_path)

    report = run_model_downloads(
        tmp_path,
        role="brain",
        apply=True,
        yes=True,
        download_func=_write_payload,
    )

    assert report.real_model_ready is False
    assert report.real_model_verified is False


def test_cli_model_download_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = types.SimpleNamespace(home=tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    captured: dict[str, Any] = {}

    def _fake_run(home: Path, **kwargs: Any) -> ModelDownloadReport:
        captured["home"] = home
        captured.update(kwargs)
        return ModelDownloadReport(
            generated_at="2026-06-28T00:00:00Z",
            mode="apply",
            applied=True,
            selected_roles=["brain"],
            entries=[],
            next_commands=[],
        )

    monkeypatch.setattr("apps.runner.main.run_model_downloads", _fake_run)

    result = CliRunner().invoke(
        app,
        [
            "april",
            "model",
            "download",
            "--role",
            "brain",
            "--apply",
            "--yes",
            "--skip-existing",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["home"] == tmp_path
    assert captured["role"] == "brain"
    assert captured["apply"] is True
    assert captured["yes"] is True
    assert captured["skip_existing"] is True


def test_cli_model_download_apply_requires_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = types.SimpleNamespace(home=tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)

    result = CliRunner().invoke(
        app,
        ["april", "model", "download", "--all-core", "--apply"],
    )

    assert result.exit_code == 1
    assert "--yes" in result.output
