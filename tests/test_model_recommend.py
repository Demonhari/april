from __future__ import annotations

import json
import types

import pytest
from typer.testing import CliRunner

from apps.runner import model_tools
from apps.runner.main import app
from april_common.settings import project_root


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


def test_cli_model_recommend_json(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = types.SimpleNamespace(home=project_root())
    monkeypatch.setattr("apps.runner.main._manager", lambda: manager)
    monkeypatch.setattr(model_tools, "machine_kind", lambda: "macOS Apple Silicon")
    result = CliRunner().invoke(app, ["april", "model", "recommend", "--json"])
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["recommended_profile"] in data["available_profiles"]
    assert data["mutating"] is False
