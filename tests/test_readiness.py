from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from apps.runner.readiness import ReadinessReport, build_readiness_report


@pytest.fixture(autouse=True)
def _clear_april_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # The builder reads process env; isolate it from any leaked APRIL_* override
    # so temp-config scenarios are deterministic.
    for key in list(os.environ):
        if key.startswith("APRIL_"):
            monkeypatch.delenv(key, raising=False)


def _model_entry(model_id: str, *, path: str, backend: str = "llama_cpp", role: str) -> dict:
    return {
        "id": model_id,
        "name": model_id,
        "path": path,
        "backend": backend,
        "role": role,
        "threads": 4,
        "context_size": 1024,
        "temperature": 0.2,
        "max_output_tokens": 256,
    }


def _write_home(
    home: Path,
    *,
    backend: str = "fake",
    models: dict[str, dict] | None = None,
    voice: dict | None = None,
) -> Path:
    configs = home / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    april: dict = {"environment": "development", "runtime": {"backend": backend}}
    if voice is not None:
        april["voice"] = voice
    (configs / "april.yaml").write_text(yaml.safe_dump(april), encoding="utf-8")
    if models is None:
        models = {
            "brain": _model_entry("april-brain", path="models/brain.gguf", role="brain"),
        }
    (configs / "models.yaml").write_text(yaml.safe_dump({"models": models}), encoding="utf-8")
    return home


def test_fake_backend_without_models_is_not_ready(tmp_path: Path) -> None:
    home = _write_home(tmp_path, backend="fake")
    report = build_readiness_report(home)
    assert isinstance(report, ReadinessReport)
    assert report.real_model_ready is False
    assert report.runtime_is_fake is True
    assert "runtime backend" in report.blockers
    assert "configured GGUF model files" in report.blockers
    # Actionable commands only, and the authoritative real-verify command is last.
    if not report.llama_cpp_python_available:
        assert "pip install -e '.[runtime]'" in report.next_actions
    assert any(action.startswith("run april verify") for action in report.next_actions)


def test_present_model_files_clear_the_gguf_blocker(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "brain.gguf").write_bytes(b"GGUF\x00fake")
    home = _write_home(
        tmp_path,
        backend="llama_cpp",
        models={"brain": _model_entry("april-brain", path="models/brain.gguf", role="brain")},
    )
    report = build_readiness_report(home)
    gguf = next(c for c in report.checks if c.name == "configured GGUF model files")
    assert gguf.status == "ok"
    assert "configured GGUF model files" not in report.blockers
    assert report.runtime_is_fake is False
    backend_check = next(c for c in report.checks if c.name == "runtime backend")
    assert backend_check.status == "ok"
    assert report.models[0].path_exists is True
    assert report.models[0].path_basename == "brain.gguf"


def test_missing_model_file_is_a_blocker(tmp_path: Path) -> None:
    home = _write_home(
        tmp_path,
        backend="llama_cpp",
        models={"brain": _model_entry("april-brain", path="models/brain.gguf", role="brain")},
    )
    report = build_readiness_report(home)
    assert "configured GGUF model files" in report.blockers
    assert report.real_model_ready is False
    assert report.models[0].path_exists is False


def test_voice_disabled_artifacts_are_skipped_not_blockers(tmp_path: Path) -> None:
    home = _write_home(tmp_path, backend="fake")
    report = build_readiness_report(home)
    assert report.voice_enabled is False
    voice_checks = [c for c in report.checks if c.name.startswith("voice:")]
    assert voice_checks
    assert all(c.status == "skipped" for c in voice_checks)
    # Voice never blocks model readiness, and disabled voice is not "ready".
    assert all(not name.startswith("voice:") for name in report.blockers)
    assert report.voice_ready is False


def test_voice_enabled_missing_artifacts_block_voice_only(tmp_path: Path) -> None:
    home = _write_home(
        tmp_path,
        backend="llama_cpp",
        voice={
            "enabled": True,
            "whisper_binary_path": "voice/whisper",
            "whisper_model_path": "voice/whisper.bin",
            "piper_binary_path": "voice/piper",
            "piper_model_path": "voice/piper.onnx",
            "wake_word_model_path": "voice/april.onnx",
        },
    )
    report = build_readiness_report(home)
    assert report.voice_enabled is True
    assert report.voice_ready is False
    voice_blockers = [name for name in report.blockers if name.startswith("voice:")]
    assert len(voice_blockers) == 5
    # A voice-only blocker must not flip real_model_ready on its own.
    model_blockers = [name for name in report.blockers if not name.startswith("voice:")]
    assert "runtime backend" not in model_blockers
    assert "run april setup voice" in report.next_actions
    assert "run april voice verify-live --report data/verification/voice-live.json" in (
        report.next_actions
    )


def test_default_development_tokens_warn_not_block(tmp_path: Path) -> None:
    home = _write_home(tmp_path, backend="fake")
    report = build_readiness_report(home)
    token_check = next(c for c in report.checks if c.name == "api/runtime tokens")
    assert token_check.status == "warning"
    assert "api/runtime tokens" in report.warnings
    assert "api/runtime tokens" not in report.blockers
    assert report.api_token_status == "default-development"
    assert "run april setup tokens" in report.next_actions


def test_blank_voice_paths_report_as_not_configured(tmp_path: Path) -> None:
    # Blank optional voice paths from .env must resolve to None, so readiness shows
    # them as "not configured" rather than as the repo root (the Path(".") bug).
    home = _write_home(tmp_path, backend="fake")
    (home / ".env").write_text(
        "\n".join(
            [
                "APRIL_WHISPER_BINARY_PATH=",
                "APRIL_WHISPER_MODEL_PATH=",
                "APRIL_PIPER_BINARY_PATH=",
                "APRIL_PIPER_MODEL_PATH=",
                "APRIL_WAKE_WORD_MODEL_PATH=",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = build_readiness_report(home)
    assert report.voice_artifacts
    assert all(not artifact.configured for artifact in report.voice_artifacts)
    assert all(not artifact.exists for artifact in report.voice_artifacts)
    # The repo-root basename must never appear as a "configured" voice artifact.
    assert all(artifact.basename is None for artifact in report.voice_artifacts)


def test_placeholder_tokens_warn_and_are_never_printed(tmp_path: Path) -> None:
    home = _write_home(tmp_path, backend="fake")
    (home / ".env").write_text(
        "APRIL_API_TOKEN=change-me-local-token\nAPRIL_RUNTIME_TOKEN=change-me-runtime-token\n",
        encoding="utf-8",
    )
    report = build_readiness_report(home)
    token_check = next(c for c in report.checks if c.name == "api/runtime tokens")
    assert token_check.status == "warning"
    assert "api/runtime tokens" in report.warnings
    assert "api/runtime tokens" not in report.blockers
    assert report.api_token_status == "placeholder-insecure"
    assert report.runtime_token_status == "placeholder-insecure"
    assert "run april setup tokens" in report.next_actions
    # The placeholder values must never be printed anywhere in the report.
    blob = json.dumps(report.model_dump())
    assert "change-me-local-token" not in blob
    assert "change-me-runtime-token" not in blob


def test_report_is_json_serialisable_and_redacted(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "brain.gguf").write_bytes(b"GGUF")
    home = _write_home(
        tmp_path,
        backend="llama_cpp",
        models={"brain": _model_entry("april-brain", path="models/brain.gguf", role="brain")},
        voice={"enabled": True, "whisper_binary_path": "voice/whisper-bin"},
    )
    report = build_readiness_report(home)
    blob = json.dumps(report.model_dump())
    # JSON output mode round-trips and never leaks absolute paths or token values.
    assert json.loads(blob)
    assert str(tmp_path) not in blob
    assert "local-dev-token" not in blob
    assert "local-dev-runtime-token" not in blob
    # Only basenames / status words survive.
    assert report.models[0].path_basename == "brain.gguf"
    assert report.api_token_status == "default-development"


def test_broken_config_reports_a_single_blocker(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "april.yaml").write_text("environment: development\n", encoding="utf-8")
    (configs / "models.yaml").write_text("models: [not, a, mapping]\n", encoding="utf-8")
    report = build_readiness_report(tmp_path)
    assert report.real_model_ready is False
    assert "model registry" in report.blockers
