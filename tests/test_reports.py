from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from apps.runner.main import app
from apps.runner.reports import (
    classify_report,
    clean_reports,
    latest_report,
    latest_report_of_type,
    list_report_summaries,
    summarize_path,
)
from april_common.settings import load_settings


class FakeManager:
    def __init__(self, home: Path) -> None:
        self.home = home
        self.settings = load_settings(root=home)


def _write_report(
    directory: Path, name: str, payload: dict[str, Any], *, age_days: float = 0.0
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    if age_days:
        past = time.time() - age_days * 86_400
        os.utime(path, (past, past))
    return path


def _acceptance(generated_at: str, final_status: str = "warning") -> dict[str, Any]:
    return {
        "report_type": "acceptance",
        "generated_at": generated_at,
        "final_status": final_status,
        "acceptance_level": "fake_sanity",
        "runtime_backend": "llama_cpp",
        "services": {"requested": True, "startup_status": "ok", "shutdown_status": "stopped"},
        "next_actions": ["run april acceptance --require-real-models"],
    }


def _activation(generated_at: str) -> dict[str, Any]:
    return {
        "report_type": "mac_activation",
        "generated_at": generated_at,
        "final_status": "applied",
        "next_actions": ["pip install -e '.[runtime]'"],
    }


def _go_live(generated_at: str, final_status: str = "pass") -> dict[str, Any]:
    return {
        "report_type": "go_live",
        "generated_at": generated_at,
        "final_status": final_status,
        "acceptance_level": "real_models",
        "runtime_backend": "llama_cpp",
        "services": {"requested": True, "startup_status": "ok", "shutdown_status": "stopped"},
        "next_actions": ["run april go-live --write-report --start-services"],
    }


# --- classification --------------------------------------------------------


def test_classify_known_and_alias_and_unknown() -> None:
    assert classify_report({"report_type": "acceptance"}) == "acceptance"
    assert classify_report({"report_type": "go_live"}) == "go_live"
    assert classify_report({"report_type": "soak"}) == "fake_soak"
    assert classify_report({"verification_level": "all", "models": []}) == "multi_model"
    assert classify_report({"iterations": 3, "latency_ms": {}}) == "fake_soak"
    assert classify_report({"something_else": 1}) == "unknown"


def test_go_live_report_is_browsable(tmp_path: Path) -> None:
    directory = tmp_path / "data" / "verification"
    _write_report(directory, "go.json", _go_live("2026-06-29T00:00:00Z"))
    # Listing recognizes the new type and surfaces final_status as the status.
    listing = list_report_summaries(directory)
    assert listing.count == 1
    summary = listing.reports[0]
    assert summary.report_type == "go_live"
    assert summary.status == "pass"
    assert summary.acceptance_level == "real_models"
    assert summary.runtime_backend == "llama_cpp"
    # latest / latest-of-type both find it.
    latest = latest_report(directory)
    assert latest is not None
    assert latest.report_type == "go_live"
    typed = latest_report_of_type(directory, "go_live")
    assert typed is not None
    assert typed.basename == "go.json"


# --- listing / latest / show-latest ----------------------------------------


def test_reports_list_newest_first(tmp_path: Path) -> None:
    directory = tmp_path / "data" / "verification"
    _write_report(directory, "a.json", _acceptance("2026-06-20T00:00:00Z"))
    _write_report(directory, "b.json", _acceptance("2026-06-28T00:00:00Z"))
    _write_report(directory, "c.json", _acceptance("2026-06-24T00:00:00Z"))
    listing = list_report_summaries(directory)
    assert [report.basename for report in listing.reports] == ["b.json", "c.json", "a.json"]
    assert listing.count == 3


def test_latest_report_returns_newest_known(tmp_path: Path) -> None:
    directory = tmp_path / "data" / "verification"
    _write_report(directory, "acc.json", _acceptance("2026-06-25T00:00:00Z"))
    # A newer file of an unknown type must not shadow the newest known report.
    _write_report(directory, "mystery.json", {"generated_at": "2026-06-29T00:00:00Z", "x": 1})
    summary = latest_report(directory)
    assert summary is not None
    assert summary.basename == "acc.json"
    assert summary.report_type == "acceptance"


def test_latest_report_of_type_filters(tmp_path: Path) -> None:
    directory = tmp_path / "data" / "verification"
    _write_report(directory, "acc-old.json", _acceptance("2026-06-20T00:00:00Z"))
    _write_report(directory, "acc-new.json", _acceptance("2026-06-28T00:00:00Z"))
    _write_report(directory, "act.json", _activation("2026-06-27T00:00:00Z"))
    acceptance = latest_report_of_type(directory, "acceptance")
    activation = latest_report_of_type(directory, "mac_activation")
    assert acceptance is not None
    assert acceptance.basename == "acc-new.json"
    assert activation is not None
    assert activation.basename == "act.json"
    assert latest_report_of_type(directory, "voice_live") is None


# --- clean -----------------------------------------------------------------


def test_clean_is_dry_run_and_lists_old_reports(tmp_path: Path) -> None:
    directory = tmp_path / "data" / "verification"
    old = _write_report(directory, "old.json", _acceptance("2026-06-01T00:00:00Z"), age_days=30)
    _write_report(directory, "new.json", _acceptance("2026-06-28T00:00:00Z"), age_days=1)
    result = clean_reports(directory, older_than_days=14, apply=False)
    assert result.applied is False
    assert [c.basename for c in result.candidates] == ["old.json"]
    assert result.deleted == []
    # Dry-run never deletes.
    assert old.exists()


def test_clean_apply_deletes_only_old_json_inside_directory(tmp_path: Path) -> None:
    directory = tmp_path / "data" / "verification"
    old = _write_report(directory, "old.json", _acceptance("2026-06-01T00:00:00Z"), age_days=30)
    new = _write_report(directory, "new.json", _acceptance("2026-06-28T00:00:00Z"), age_days=1)
    # A non-JSON file and an outside file must never be touched.
    note = directory / "notes.txt"
    note.write_text("keep me", encoding="utf-8")
    os.utime(note, (time.time() - 60 * 86_400, time.time() - 60 * 86_400))
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    os.utime(outside, (time.time() - 60 * 86_400, time.time() - 60 * 86_400))

    result = clean_reports(directory, older_than_days=14, apply=True)
    assert result.applied is True
    assert result.deleted == ["old.json"]
    assert not old.exists()
    assert new.exists()
    assert note.exists()
    assert outside.exists()


# --- redaction -------------------------------------------------------------


def test_summary_drops_non_whitelisted_secret_fields(tmp_path: Path) -> None:
    directory = tmp_path / "data" / "verification"
    payload = _acceptance("2026-06-28T00:00:00Z")
    payload["api_token"] = "tok-super-secret"
    payload["transcript"] = "the user said something private"
    payload["generated_text"] = "model wrote this"
    payload["next_actions"] = ["fix model at /Users/secret/models/brain.gguf"]
    path = _write_report(directory, "acc.json", payload)
    summary = summarize_path(path)
    assert summary is not None
    dumped = json.dumps(summary.model_dump())
    assert "tok-super-secret" not in dumped
    assert "the user said something private" not in dumped
    assert "model wrote this" not in dumped
    # Absolute paths in next actions are reduced to basenames.
    assert "/Users/secret/models" not in dumped
    assert "brain.gguf" in dumped


# --- CLI -------------------------------------------------------------------


def test_reports_cli_list_and_show_latest(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    directory = tmp_path / "data" / "verification"
    _write_report(directory, "acc.json", _acceptance("2026-06-28T00:00:00Z"))
    _write_report(directory, "act.json", _activation("2026-06-27T00:00:00Z"))
    runner = CliRunner()
    listed = runner.invoke(app, ["april", "reports", "list"])
    assert listed.exit_code == 0
    assert "acc.json" in listed.output
    shown = runner.invoke(app, ["april", "reports", "show-latest", "--type", "mac_activation"])
    assert shown.exit_code == 0
    assert "act.json" in shown.output


def test_reports_cli_show_latest_unknown_type_errors(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    result = CliRunner().invoke(app, ["april", "reports", "show-latest", "--type", "bogus"])
    assert result.exit_code == 1


def test_reports_cli_clean_dry_run_default(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    directory = tmp_path / "data" / "verification"
    old = _write_report(directory, "old.json", _acceptance("2026-06-01T00:00:00Z"), age_days=30)
    result = CliRunner().invoke(app, ["april", "reports", "clean", "--older-than-days", "14"])
    assert result.exit_code == 0
    assert "Would delete" in result.output
    assert old.exists()


def test_reports_cli_clean_apply_deletes(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    directory = tmp_path / "data" / "verification"
    old = _write_report(directory, "old.json", _acceptance("2026-06-01T00:00:00Z"), age_days=30)
    result = CliRunner().invoke(
        app, ["april", "reports", "clean", "--older-than-days", "14", "--apply"]
    )
    assert result.exit_code == 0
    assert not old.exists()


def test_reports_cli_show_redacts_secret_fields(tmp_path: Path, monkeypatch) -> None:
    manager = FakeManager(tmp_path)
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    payload = _acceptance("2026-06-28T00:00:00Z")
    payload["api_token"] = "tok-super-secret"
    path = _write_report(tmp_path, "explicit.json", payload)
    result = CliRunner().invoke(app, ["april", "reports", "show", str(path), "--json"])
    assert result.exit_code == 0
    assert "tok-super-secret" not in result.output
