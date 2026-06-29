from __future__ import annotations

import json
import shutil
from datetime import timedelta
from pathlib import Path

import yaml

from apps.runner.daily_driver import build_daily_driver_report
from april_common.config_fingerprint import config_fingerprint_digest
from april_common.time import utc_now


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


def _iso(days_ago: float) -> str:
    return (utc_now() - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_report(home: Path, basename: str, payload: dict) -> None:
    reports = home / "data" / "verification"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / basename).write_text(json.dumps(payload), encoding="utf-8")


def _ready_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    _copy_configs(home)
    _create_gguf_files(home)
    return home


def test_daily_driver_redacted_and_schema(tmp_path: Path) -> None:
    home = _ready_home(tmp_path)
    report = build_daily_driver_report(home)
    blob = json.dumps(report.model_dump())
    assert "/Users" not in blob
    assert str(home) not in blob
    assert "local-dev-token" not in blob
    assert report.config_fingerprint
    # All 14 daily-driver checks are present.
    names = {check.name for check in report.checks}
    for expected in (
        "config validation",
        "runtime backend",
        "llama-cpp-python",
        "configured GGUF presence",
        "latest real-model verification",
        "latest workflow-real verification",
        "latest go-live",
        "token hardening",
        "embedding provider",
        "vector index compatibility",
        "voice milestone",
        "desktop readiness",
        "report directory",
        "audit log",
    ):
        assert expected in names


def test_core_ready_with_fresh_matching_real_report(
    tmp_path: Path, llama_cpp_available: None
) -> None:
    home = _ready_home(tmp_path)
    fingerprint = config_fingerprint_digest(home)
    _write_report(
        home,
        "mac-readiness.json",
        {
            "report_type": "multi_model",
            "generated_at": _iso(1),
            "config_fingerprint": fingerprint,
            "summary": "pass",
            "real_model_verified": True,
            "verification_level": "all",
            "models": [],
        },
    )
    report = build_daily_driver_report(home)
    assert report.core_real_model == "ready"


def test_core_not_run_without_real_report(tmp_path: Path, llama_cpp_available: None) -> None:
    home = _ready_home(tmp_path)
    report = build_daily_driver_report(home)
    assert report.core_real_model == "not_run"
    real_check = next(c for c in report.checks if c.name == "latest real-model verification")
    assert real_check.status == "not_run"


def test_core_blocker_when_real_report_failed(tmp_path: Path, llama_cpp_available: None) -> None:
    home = _ready_home(tmp_path)
    _write_report(
        home,
        "mac-readiness.json",
        {
            "report_type": "multi_model",
            "generated_at": _iso(1),
            "summary": "fail",
            "verification_level": "none",
            "models": [],
        },
    )
    report = build_daily_driver_report(home)
    assert report.core_real_model == "blocker"


def test_core_blocker_when_gguf_missing(tmp_path: Path, llama_cpp_available: None) -> None:
    # Runtime extra present, but no configured GGUF files exist on disk: the core
    # real-model path is still blocked on the missing model files.
    home = tmp_path / "home"
    home.mkdir()
    _copy_configs(home)  # no _create_gguf_files → files absent
    report = build_daily_driver_report(home)
    assert report.core_real_model == "blocker"
    gguf_check = next(c for c in report.checks if c.name == "configured GGUF presence")
    assert gguf_check.status == "blocker"
    assert gguf_check.next_command == "run april setup models"


def test_stale_report_marks_warning_by_age(tmp_path: Path, llama_cpp_available: None) -> None:
    home = _ready_home(tmp_path)
    _write_report(
        home,
        "mac-readiness.json",
        {
            "report_type": "multi_model",
            "generated_at": _iso(10),  # older than the 7-day TTL
            "summary": "pass",
            "real_model_verified": True,
            "verification_level": "all",
            "models": [],
        },
    )
    report = build_daily_driver_report(home)
    assert report.core_real_model == "warning"
    real_check = next(c for c in report.checks if c.name == "latest real-model verification")
    assert "stale" in real_check.detail


def test_fingerprint_mismatch_marks_report_stale(tmp_path: Path, llama_cpp_available: None) -> None:
    home = _ready_home(tmp_path)
    _write_report(
        home,
        "mac-readiness.json",
        {
            "report_type": "multi_model",
            "generated_at": _iso(0.1),
            "config_fingerprint": "stale-digest-value",
            "summary": "pass",
            "real_model_verified": True,
            "verification_level": "all",
            "models": [],
        },
    )
    report = build_daily_driver_report(home)
    assert report.core_real_model == "warning"
    real_check = next(c for c in report.checks if c.name == "latest real-model verification")
    assert "config changed" in (real_check.detail or "")


def test_config_blocker_makes_overall_blocker(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    _copy_configs(home)
    # Corrupt models.yaml so config validation fails.
    (home / "configs" / "models.yaml").write_text("models: [not-a-mapping]\n", encoding="utf-8")
    report = build_daily_driver_report(home)
    assert report.overall == "blocker"
    config_check = next(c for c in report.checks if c.name == "config validation")
    assert config_check.status == "blocker"


def test_token_and_embedding_warnings_surface(tmp_path: Path) -> None:
    home = _ready_home(tmp_path)
    report = build_daily_driver_report(home)
    # Default-development tokens and hashed-token embeddings are the hardened-rung
    # reasons, surfaced without ever blocking the core path.
    assert report.hardened_reason is not None
    assert "development tokens" in report.hardened_reason
    assert "hashed-token embeddings" in report.hardened_reason
    token_check = next(c for c in report.checks if c.name == "token hardening")
    assert token_check.status == "warning"
    assert token_check.next_command == "run april setup tokens"
