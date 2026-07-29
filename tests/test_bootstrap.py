from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from apps.runner.bootstrap import bootstrap
from apps.runner.main import app


@pytest.fixture
def home_with_configs(settings_tmp) -> Path:
    home = settings_tmp.home
    shutil.copytree(Path.cwd() / "configs", home / "configs")
    models_path = home / "configs" / "models.yaml"
    data = yaml.safe_load(models_path.read_text(encoding="utf-8"))
    for model in data["models"].values():
        model["path"] = str(home / "models" / f"{model['id']}.gguf")
    models_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    april_path = home / "configs" / "april.yaml"
    april = yaml.safe_load(april_path.read_text(encoding="utf-8"))
    voice = april.setdefault("voice", {})
    voice.update(
        {
            "enabled": False,
            "whisper_binary_path": None,
            "whisper_model_path": None,
            "piper_binary_path": None,
            "piper_model_path": None,
            "wake_word_model_path": None,
        }
    )
    april_path.write_text(yaml.safe_dump(april, sort_keys=False), encoding="utf-8")
    return home


def test_bootstrap_creates_directories(home_with_configs: Path) -> None:
    report = bootstrap(home_with_configs)
    for relative in (
        "data",
        "logs",
        "models",
        "data/run",
        "data/artifacts/patches",
        "data/artifacts/cleanup",
        "data/audio_cache",
        "data/vector_index",
    ):
        assert (home_with_configs / relative).is_dir(), relative
    assert any(item["created"] for item in report["directories"])


def test_bootstrap_is_non_destructive_on_rerun(home_with_configs: Path) -> None:
    bootstrap(home_with_configs)
    rerun = bootstrap(home_with_configs)
    # Second run creates nothing new and keeps the existing tokens.
    assert all(item["created"] is False for item in rerun["directories"])
    assert rerun["tokens"]["action"] == "kept"


def test_bootstrap_generates_tokens_without_printing(home_with_configs: Path) -> None:
    report = bootstrap(home_with_configs)
    env_file = home_with_configs / ".env"
    assert env_file.exists()
    content = env_file.read_text(encoding="utf-8")
    assert "APRIL_API_TOKEN=" not in content
    assert "APRIL_RUNTIME_TOKEN=" not in content
    assert "APRIL_API_CREDENTIAL_ID=core-api-token" in content
    assert "APRIL_RUNTIME_CREDENTIAL_ID=runtime-auth-token" in content
    assert report["tokens"]["action"] == "generated"
    assert report["tokens"]["api_token_set"] is True
    assert report["tokens"]["runtime_token_set"] is True


def test_bootstrap_keeps_existing_tokens_without_force(home_with_configs: Path) -> None:
    env_file = home_with_configs / ".env"
    env_file.write_text(
        "APRIL_API_TOKEN=existing-api\nAPRIL_RUNTIME_TOKEN=existing-runtime\n", encoding="utf-8"
    )
    report = bootstrap(home_with_configs)
    assert report["tokens"]["action"] == "migration_required"
    assert "existing-api" in env_file.read_text(encoding="utf-8")


def test_bootstrap_force_regenerates_tokens(home_with_configs: Path) -> None:
    env_file = home_with_configs / ".env"
    env_file.write_text(
        "APRIL_API_TOKEN=existing-api\nAPRIL_RUNTIME_TOKEN=existing-runtime\n", encoding="utf-8"
    )
    report = bootstrap(home_with_configs, force=True)
    assert report["tokens"]["action"] == "migration_required"
    assert "existing-api" in env_file.read_text(encoding="utf-8")


def test_bootstrap_recommends_profile_without_applying(home_with_configs: Path) -> None:
    report = bootstrap(home_with_configs, no_auto_profile=True)
    assert report["recommended_profile"]
    assert report["profile_applied"] is False
    assert report["applied_profile"] is None
    assert report["auto_profile_suppressed"] is True


def test_bootstrap_applies_profile_only_with_flag(home_with_configs: Path) -> None:
    report = bootstrap(home_with_configs, apply_profile=True)
    assert report["profile_applied"] is True
    assert report["applied_profile"] == report["recommended_profile"]


def test_bootstrap_auto_applies_intel_profile_once(
    home_with_configs: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("apps.runner.bootstrap.platform.system", lambda: "Darwin")
    monkeypatch.setattr("apps.runner.bootstrap.platform.machine", lambda: "x86_64")
    monkeypatch.setattr(
        "apps.runner.bootstrap.recommend_model_profile",
        lambda _home: {
            "recommended_profile": "intel_macbook_cpu_low",
            "available_profiles": ["intel_macbook_cpu_low"],
            "expected_backend": "CPU-only",
            "architecture": "x86_64",
            "platform": "Darwin",
            "cpu_count": 8,
            "available_memory": None,
            "arm64_python": False,
        },
    )
    report = bootstrap(home_with_configs)
    assert report["profile_auto_applied"] is True
    rerun = bootstrap(home_with_configs)
    assert rerun["profile_auto_applied"] is False


def test_bootstrap_preserves_manual_runtime_override(
    home_with_configs: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_path = home_with_configs / "configs" / "models.yaml"
    data = yaml.safe_load(models_path.read_text(encoding="utf-8"))
    first = next(iter(data["models"].values()))
    first["threads"] = 13
    models_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr("apps.runner.bootstrap.platform.system", lambda: "Darwin")
    monkeypatch.setattr("apps.runner.bootstrap.platform.machine", lambda: "x86_64")
    monkeypatch.setattr(
        "apps.runner.bootstrap.recommend_model_profile",
        lambda _home: {
            "recommended_profile": "intel_macbook_cpu_low",
            "available_profiles": ["intel_macbook_cpu_low"],
            "expected_backend": "CPU-only",
            "architecture": "x86_64",
            "platform": "Darwin",
            "cpu_count": 8,
            "available_memory": None,
            "arm64_python": False,
        },
    )
    report = bootstrap(home_with_configs)
    assert report["profile_applied"] is False
    updated = yaml.safe_load(models_path.read_text(encoding="utf-8"))
    assert next(iter(updated["models"].values()))["threads"] == 13


def test_bootstrap_reports_models_voice_roots_and_validation(home_with_configs: Path) -> None:
    report = bootstrap(home_with_configs)
    assert isinstance(report["llama_cpp_available"], bool)
    assert report["models"]  # configured models are reported
    # The configured GGUF files do not exist in a fresh home.
    assert report["missing_model_paths"]
    assert "paths" in report["voice"]
    assert report["allowed_filesystem_roots"]
    assert report["config_valid"] is True
    assert any("verify --fake" in command for command in report["next_commands"])


def test_bootstrap_warns_about_dev_tokens_when_env_not_loaded(home_with_configs: Path) -> None:
    # Secure defaults do not retain development tokens in configuration.
    side_env = home_with_configs / "side.env"
    report = bootstrap(home_with_configs, env_file=side_env)
    assert side_env.exists()
    assert report["dev_token_warnings"] == []


def test_bootstrap_warns_for_placeholder_tokens_without_printing_values(
    home_with_configs: Path,
) -> None:
    config = home_with_configs / "configs" / "april.yaml"
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data.setdefault("api", {})["token"] = "change-me-local-token"
    data.setdefault("runtime", {})["token"] = "change-me-runtime-token"
    config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = bootstrap(home_with_configs, env_file=home_with_configs / "side.env")

    warnings = " ".join(report["dev_token_warnings"])
    assert "placeholder" in warnings
    assert report["tokens"]["action"] == "migration_required"
    blob = json.dumps(report)
    assert "change-me-local-token" not in blob
    assert "change-me-runtime-token" not in blob


def test_bootstrap_warns_for_blank_or_missing_tokens(
    home_with_configs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APRIL_API_TOKEN", "")
    monkeypatch.setenv("APRIL_RUNTIME_TOKEN", "")

    report = bootstrap(home_with_configs, env_file=home_with_configs / "side.env")

    assert report["dev_token_warnings"] == []
    assert report["tokens"]["action"] == "generated"


def test_bootstrap_output_does_not_contain_existing_token_values(home_with_configs: Path) -> None:
    env_file = home_with_configs / ".env"
    env_file.write_text(
        "APRIL_API_TOKEN=existing-api-secret\nAPRIL_RUNTIME_TOKEN=existing-runtime-secret\n",
        encoding="utf-8",
    )

    report = bootstrap(home_with_configs)

    blob = json.dumps(report)
    assert "existing-api-secret" not in blob
    assert "existing-runtime-secret" not in blob


def test_bootstrap_voice_relative_path_resolves_under_april_home(
    home_with_configs: Path,
) -> None:
    voice_binary = home_with_configs / "voice" / "whisper-main"
    voice_binary.parent.mkdir()
    voice_binary.write_text("#!/bin/sh\n", encoding="utf-8")
    config = home_with_configs / "configs" / "april.yaml"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "  whisper_binary_path: null", "  whisper_binary_path: voice/whisper-main"
        ),
        encoding="utf-8",
    )

    report = bootstrap(home_with_configs)

    whisper = next(item for item in report["voice"]["paths"] if item["name"] == "whisper_binary")
    assert whisper["configured"] is True
    assert whisper["exists"] is True
    assert whisper["path"] == "whisper-main"


def test_bootstrap_voice_relative_path_does_not_use_current_working_directory(
    home_with_configs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    cwd_voice = outside / "voice" / "whisper-main"
    cwd_voice.parent.mkdir(parents=True)
    cwd_voice.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.chdir(outside)
    config = home_with_configs / "configs" / "april.yaml"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "  whisper_binary_path: null", "  whisper_binary_path: voice/whisper-main"
        ),
        encoding="utf-8",
    )

    report = bootstrap(home_with_configs)

    whisper = next(item for item in report["voice"]["paths"] if item["name"] == "whisper_binary")
    assert whisper["configured"] is True
    assert whisper["exists"] is False


def test_setup_bootstrap_cli(home_with_configs: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "apps.runner.main._manager", lambda: SimpleNamespace(home=home_with_configs)
    )
    result = CliRunner().invoke(app, ["april", "setup", "bootstrap", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["config_valid"] is True
    assert (home_with_configs / ".env").exists()
    assert str(home_with_configs) not in result.output


def test_setup_bootstrap_cli_show_paths_opt_in(
    home_with_configs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "apps.runner.main._manager", lambda: SimpleNamespace(home=home_with_configs)
    )

    result = CliRunner().invoke(app, ["april", "setup", "bootstrap", "--json", "--show-paths"])

    assert result.exit_code == 0, result.output
    assert str(home_with_configs) in result.output
