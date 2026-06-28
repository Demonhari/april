from __future__ import annotations

import json
import shutil
import types
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from apps.runner import model_tools
from apps.runner.main import app
from april_common.errors import ConfigError
from april_common.settings import project_root


def _copy_configs(home: Path) -> None:
    shutil.copytree(project_root() / "configs", home / "configs")


def _add_reasoning_model(home: Path) -> None:
    path = home / "configs" / "models.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["models"]["reasoning"] = {
        "id": "april-reasoning",
        "name": "reasoning",
        "path": "models/reasoning.gguf",
        "backend": "llama_cpp",
        "role": "reasoning",
        "threads": 1,
        "context_size": 1024,
        "temperature": 0.1,
        "max_output_tokens": 256,
        "keep_loaded": True,
        "idle_unload_seconds": 60,
        "priority": 60,
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_apple_silicon_recommendation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_tools, "machine_kind", lambda: "macOS Apple Silicon")
    monkeypatch.setattr(model_tools.platform, "machine", lambda: "arm64")
    payload = model_tools.recommend_model_profile(project_root())
    assert payload["architecture"] == "macOS Apple Silicon"
    assert payload["recommended_profile"] == "apple_silicon_macbook"
    assert "Metal" in payload["expected_backend"]
    assert payload["arm64_python"] is True
    assert payload["mutating"] is False
    assert payload["recommended_profile"] in payload["available_profiles"]
    assert payload["cpu_count"]


def test_intel_recommendation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_tools, "machine_kind", lambda: "macOS Intel")
    monkeypatch.setattr(model_tools.platform, "machine", lambda: "x86_64")
    payload = model_tools.recommend_model_profile(project_root())
    assert payload["recommended_profile"] == "intel_macbook_cpu_low"
    assert "CPU" in payload["expected_backend"]
    assert payload["arm64_python"] is False


def test_non_arm64_apple_silicon_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_tools, "machine_kind", lambda: "macOS Apple Silicon")
    monkeypatch.setattr(model_tools.platform, "machine", lambda: "x86_64")
    payload = model_tools.recommend_model_profile(project_root())
    assert any("arm64" in note.lower() for note in payload["notes"])


def test_recommendation_is_non_mutating() -> None:
    models_yaml = project_root() / "configs" / "models.yaml"
    before = models_yaml.read_text(encoding="utf-8")
    model_tools.recommend_model_profile(project_root())
    assert models_yaml.read_text(encoding="utf-8") == before


def test_profiles_define_intel_and_apple_silicon() -> None:
    profiles = model_tools.load_model_profiles(project_root())
    assert "intel_macbook_cpu_low" in profiles
    assert "apple_silicon_macbook" in profiles

    apple = profiles["apple_silicon_macbook"]
    assert apple["brain"]["n_gpu_layers"] == -1  # Metal offload
    assert apple["brain"]["keep_loaded"] is True
    assert apple["coding"]["idle_unload_seconds"]  # specialist eviction policy

    intel = profiles["intel_macbook_cpu_low"]
    assert intel["brain"]["n_gpu_layers"] == 0  # CPU-only, no Metal
    assert intel["brain"]["keep_loaded"] is True  # one always-loaded brain
    assert intel["coding"]["keep_loaded"] is False  # specialists on demand
    assert "reasoning" in intel
    assert "reasoning" in apple


def test_intel_profile_updates_reasoning_when_configured(tmp_path: Path) -> None:
    _copy_configs(tmp_path)
    _add_reasoning_model(tmp_path)

    model_tools.apply_model_profile(home=tmp_path, profile_name="intel_macbook_cpu_low")

    data = yaml.safe_load((tmp_path / "configs" / "models.yaml").read_text(encoding="utf-8"))
    reasoning = data["models"]["reasoning"]
    assert reasoning["context_size"] == 4096
    assert reasoning["threads"] == 8
    assert reasoning["n_batch"] == 128
    assert reasoning["n_gpu_layers"] == 0
    assert reasoning["temperature"] == 0.4
    assert reasoning["keep_loaded"] is False


def test_apple_silicon_profile_updates_reasoning_when_configured(tmp_path: Path) -> None:
    _copy_configs(tmp_path)
    _add_reasoning_model(tmp_path)

    model_tools.apply_model_profile(home=tmp_path, profile_name="apple_silicon_macbook")

    data = yaml.safe_load((tmp_path / "configs" / "models.yaml").read_text(encoding="utf-8"))
    reasoning = data["models"]["reasoning"]
    assert reasoning["context_size"] == 8192
    assert reasoning["threads"] == 6
    assert reasoning["n_batch"] == 256
    assert reasoning["n_gpu_layers"] == -1
    assert reasoning["temperature"] == 0.4
    assert reasoning["keep_loaded"] is False


@pytest.mark.parametrize("profile_name", ["intel_macbook_cpu_low", "apple_silicon_macbook"])
def test_profile_apply_without_reasoning_configured_still_passes(
    tmp_path: Path, profile_name: str
) -> None:
    _copy_configs(tmp_path)

    model_tools.apply_model_profile(home=tmp_path, profile_name=profile_name)

    data = yaml.safe_load((tmp_path / "configs" / "models.yaml").read_text(encoding="utf-8"))
    assert "reasoning" not in data["models"]


def test_profile_unknown_fields_are_ignored_for_reasoning(tmp_path: Path) -> None:
    _copy_configs(tmp_path)
    _add_reasoning_model(tmp_path)
    profile_path = tmp_path / "configs" / "model_profiles.yaml"
    data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    data["profiles"]["intel_macbook_cpu_low"]["reasoning"]["made_up_field"] = "ignored"
    profile_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    model_tools.apply_model_profile(home=tmp_path, profile_name="intel_macbook_cpu_low")

    models = yaml.safe_load((tmp_path / "configs" / "models.yaml").read_text(encoding="utf-8"))
    assert "made_up_field" not in models["models"]["reasoning"]


def test_profile_invalid_runtime_fields_are_rejected_and_rolled_back(tmp_path: Path) -> None:
    _copy_configs(tmp_path)
    _add_reasoning_model(tmp_path)
    models_path = tmp_path / "configs" / "models.yaml"
    before = models_path.read_bytes()
    profile_path = tmp_path / "configs" / "model_profiles.yaml"
    data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    data["profiles"]["intel_macbook_cpu_low"]["reasoning"]["threads"] = 0
    profile_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError):
        model_tools.apply_model_profile(home=tmp_path, profile_name="intel_macbook_cpu_low")

    assert models_path.read_bytes() == before


def test_cli_model_recommend_json(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = types.SimpleNamespace(home=project_root())
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr(model_tools, "machine_kind", lambda: "macOS Apple Silicon")
    result = CliRunner().invoke(app, ["april", "model", "recommend", "--json"])
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["recommended_profile"] in data["available_profiles"]
    assert data["mutating"] is False
