from __future__ import annotations

import json
from pathlib import Path

import yaml

from april_common.config_fingerprint import (
    build_config_fingerprint,
    config_fingerprint_digest,
    config_fingerprint_for_home,
)
from april_common.settings import load_settings


def _copy_configs(home: Path) -> None:
    import shutil

    shutil.copytree(Path.cwd() / "configs", home / "configs")


def test_fingerprint_is_redacted_and_structural(tmp_path: Path) -> None:
    _copy_configs(tmp_path)
    fingerprint = config_fingerprint_for_home(tmp_path)
    assert fingerprint is not None
    blob = json.dumps(fingerprint.model_dump())
    # No secrets, absolute paths, or usernames ever appear. ("hashed-token" is a
    # provider *name*, not a secret, so we assert against real token values.)
    assert "/Users" not in blob
    assert str(tmp_path) not in blob
    settings = load_settings(root=tmp_path)
    assert settings.api.token not in blob
    if settings.runtime.token:
        assert settings.runtime.token not in blob
    # Structural fields are present.
    assert fingerprint.runtime_backend
    assert fingerprint.embedding_provider in {"hashed-token", "runtime-local"}
    assert fingerprint.models
    assert all(
        model.path_basename and "/" not in model.path_basename for model in fingerprint.models
    )
    assert len(fingerprint.digest) == 16


def test_fingerprint_changes_when_structural_config_changes(tmp_path: Path) -> None:
    _copy_configs(tmp_path)
    before = config_fingerprint_digest(tmp_path)
    config = tmp_path / "configs" / "april.yaml"
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data.setdefault("memory", {})["embedding_provider"] = "runtime-local"
    data["memory"]["embedding_model_id"] = "april-embedding"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    after = config_fingerprint_digest(tmp_path)
    assert before is not None
    assert after is not None
    assert before != after


def test_fingerprint_is_stable_for_unchanged_config(tmp_path: Path) -> None:
    _copy_configs(tmp_path)
    first = config_fingerprint_digest(tmp_path)
    second = config_fingerprint_digest(tmp_path)
    assert first == second


def test_fingerprint_does_not_change_on_token_rotation(tmp_path: Path) -> None:
    # Rotating a secret token is not a structural change; the fingerprint must be
    # token-independent so a recent report is not falsely marked stale.
    _copy_configs(tmp_path)
    before = config_fingerprint_digest(tmp_path)
    config = tmp_path / "configs" / "april.yaml"
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data.setdefault("api", {})["token"] = "some-rotated-secret-token"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    after = config_fingerprint_digest(tmp_path)
    assert before == after


def test_build_fingerprint_from_settings(tmp_path: Path) -> None:
    _copy_configs(tmp_path)
    settings = load_settings(root=tmp_path)
    fingerprint = build_config_fingerprint(settings)
    assert fingerprint.digest == config_fingerprint_digest(tmp_path)
