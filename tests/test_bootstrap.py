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
    assert "APRIL_API_TOKEN=" in content
    assert "APRIL_RUNTIME_TOKEN=" in content
    assert "local-dev-token" not in content  # real tokens, not the dev defaults
    assert report["tokens"]["action"] == "generated"
    # The report must never carry the actual token values.
    api_value = next(
        line.split("=", 1)[1]
        for line in content.splitlines()
        if line.startswith("APRIL_API_TOKEN=")
    )
    assert api_value not in json.dumps(report)


def test_bootstrap_keeps_existing_tokens_without_force(home_with_configs: Path) -> None:
    env_file = home_with_configs / ".env"
    env_file.write_text(
        "APRIL_API_TOKEN=existing-api\nAPRIL_RUNTIME_TOKEN=existing-runtime\n", encoding="utf-8"
    )
    report = bootstrap(home_with_configs)
    assert report["tokens"]["action"] == "kept"
    assert "existing-api" in env_file.read_text(encoding="utf-8")


def test_bootstrap_force_regenerates_tokens(home_with_configs: Path) -> None:
    env_file = home_with_configs / ".env"
    env_file.write_text(
        "APRIL_API_TOKEN=existing-api\nAPRIL_RUNTIME_TOKEN=existing-runtime\n", encoding="utf-8"
    )
    report = bootstrap(home_with_configs, force=True)
    assert report["tokens"]["action"] == "regenerated"
    assert "existing-api" not in env_file.read_text(encoding="utf-8")


def test_bootstrap_recommends_profile_without_applying(home_with_configs: Path) -> None:
    report = bootstrap(home_with_configs)
    assert report["recommended_profile"]
    assert report["profile_applied"] is False
    assert report["applied_profile"] is None


def test_bootstrap_applies_profile_only_with_flag(home_with_configs: Path) -> None:
    report = bootstrap(home_with_configs, apply_profile=True)
    assert report["profile_applied"] is True
    assert report["applied_profile"] == report["recommended_profile"]


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
    # Write tokens to a side env file that load_settings will not read, so the
    # effective config still uses the development tokens from configs/april.yaml.
    side_env = home_with_configs / "side.env"
    report = bootstrap(home_with_configs, env_file=side_env)
    assert side_env.exists()
    assert report["dev_token_warnings"]  # effective config still on dev tokens


def test_bootstrap_warns_for_placeholder_tokens_without_printing_values(
    home_with_configs: Path,
) -> None:
    config = home_with_configs / "configs" / "april.yaml"
    text = config.read_text(encoding="utf-8")
    text = text.replace("local-dev-token", "change-me-local-token")
    text = text.replace("local-dev-runtime-token", "change-me-runtime-token")
    config.write_text(text, encoding="utf-8")

    report = bootstrap(home_with_configs, env_file=home_with_configs / "side.env")

    warnings = " ".join(report["dev_token_warnings"])
    assert "placeholder" in warnings
    blob = json.dumps(report)
    assert "change-me-local-token" not in blob
    assert "change-me-runtime-token" not in blob


def test_bootstrap_warns_for_blank_or_missing_tokens(
    home_with_configs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APRIL_API_TOKEN", "")
    monkeypatch.setenv("APRIL_RUNTIME_TOKEN", "")

    report = bootstrap(home_with_configs, env_file=home_with_configs / "side.env")

    warnings = " ".join(report["dev_token_warnings"])
    assert "APRIL_API_TOKEN is blank" in warnings
    assert "APRIL_RUNTIME_TOKEN is blank or missing" in warnings


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
