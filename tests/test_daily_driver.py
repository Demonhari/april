from __future__ import annotations

import json
import shutil
from datetime import timedelta
from pathlib import Path

import pytest
import yaml

from apps.runner.daily_driver import _vector_index_check, build_daily_driver_report
from apps.runner.readiness import ReadinessReport
from april_common.config_fingerprint import config_fingerprint_digest
from april_common.settings import load_settings
from april_common.time import utc_now
from services.memory.schemas import VectorMetadata
from services.memory.vector_memory import VectorMemory

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


def test_vector_repair_is_blocked_until_audit_is_verified(tmp_path: Path) -> None:
    home = _ready_home(tmp_path)
    settings = load_settings(root=home)
    vector = VectorMemory(settings.vector_index_path)
    metadata = VectorMetadata(
        source_type="test",
        source_id="test",
        content_hash="hash",
        created_at="2026-01-01T00:00:00Z",
    )
    vector.upsert(record_id="one", content="first", metadata=metadata)
    vector.upsert(record_id="two", content="second", metadata=metadata)
    active = vector.health()["effective_generation"]
    records = settings.vector_index_path / "generations" / active / "records.json"
    records.write_text(
        records.read_text(encoding="utf-8").replace("second", "tampered"),
        encoding="utf-8",
    )

    check = _vector_index_check(home, settings, audit_status="corrupt")

    assert check.status == "blocker"
    assert check.next_command == "run april audit verify"
    assert "before repair" in check.detail


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
            "runtime_backend": "llama_cpp",
            "real_model_verified": True,
            "real_models_exercised": 3,
            "core_model_set_verified": True,
            "all_configured_models_verified": True,
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
            "runtime_backend": "llama_cpp",
            "real_model_verified": True,
            "real_models_exercised": 3,
            "core_model_set_verified": True,
            "all_configured_models_verified": True,
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
            "runtime_backend": "llama_cpp",
            "real_model_verified": True,
            "real_models_exercised": 3,
            "core_model_set_verified": True,
            "all_configured_models_verified": True,
            "verification_level": "all",
            "models": [],
        },
    )
    report = build_daily_driver_report(home)
    assert report.core_real_model == "warning"
    real_check = next(c for c in report.checks if c.name == "latest real-model verification")
    assert "config changed" in (real_check.detail or "")


def test_simulated_report_cannot_make_daily_driver_ready(
    tmp_path: Path, llama_cpp_available: None
) -> None:
    home = _ready_home(tmp_path)
    _write_report(
        home,
        "mac-readiness.json",
        {
            "report_type": "multi_model",
            "generated_at": _iso(0.1),
            "config_fingerprint": config_fingerprint_digest(home),
            "summary": "pass",
            "runtime_backend": "fake",
            "real_model_verified": True,
            "real_models_exercised": 3,
            "core_model_set_verified": True,
            "all_configured_models_verified": True,
            "verification_level": "all",
        },
    )

    report = build_daily_driver_report(home)

    assert report.core_real_model == "warning"
    check = next(c for c in report.checks if c.name == "latest real-model verification")
    assert "not production evidence" in check.detail


def test_invalid_gguf_header_blocks_daily_driver(tmp_path: Path, llama_cpp_available: None) -> None:
    home = _ready_home(tmp_path)
    (home / "models" / "granite3.3-2b-q4_k_m.gguf").write_bytes(b"not a GGUF")

    report = build_daily_driver_report(home)

    assert report.core_real_model == "blocker"
    check = next(c for c in report.checks if c.name == "configured GGUF presence")
    assert check.status == "blocker"


def test_database_maintenance_failure_blocks_daily_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _ready_home(tmp_path)
    readiness = ReadinessReport(
        generated_at=_iso(0),
        os="Darwin",
        cpu_architecture="x86_64",
        python_version="3.12",
        runtime_backend="llama_cpp",
        runtime_is_fake=False,
        llama_cpp_python_available=True,
        environment="development",
        voice_enabled=False,
        database_quick_check="ok",
        database_foreign_key_consistent=True,
        database_wal_state="wal",
        database_integrity_failures=["migration_mismatch"],
    )
    monkeypatch.setattr(
        "apps.runner.daily_driver.build_readiness_report",
        lambda _home: readiness,
    )

    report = build_daily_driver_report(home)

    check = next(item for item in report.checks if item.name == "database integrity")
    assert check.status == "blocker"
    assert "migration_mismatch" in check.detail


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
