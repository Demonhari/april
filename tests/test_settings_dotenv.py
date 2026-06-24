from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from apps.runner.service_manager import AprilServiceManager
from april_common.settings import load_settings, reset_settings_cache
from april_common.token_setup import GeneratedTokens, write_token_env_file


def _isolate(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    monkeypatch.setenv("APRIL_ENV", "test")
    monkeypatch.setenv("APRIL_HOME", str(home))
    for key in ("APRIL_API_TOKEN", "APRIL_RUNTIME_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    reset_settings_cache()


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_process_env_overrides_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    _write(tmp_path / ".env", "APRIL_API_TOKEN=from-dotenv\n")
    monkeypatch.setenv("APRIL_API_TOKEN", "from-process-env")

    settings = load_settings(root=tmp_path)

    assert settings.api.token == "from-process-env"


def test_dotenv_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    _write(tmp_path / "configs" / "april.yaml", "api:\n  token: from-yaml\n")
    _write(tmp_path / ".env", "APRIL_API_TOKEN=from-dotenv\n")

    settings = load_settings(root=tmp_path)

    assert settings.api.token == "from-dotenv"


def test_yaml_overrides_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    _write(tmp_path / "configs" / "april.yaml", "api:\n  token: from-yaml\n")

    settings = load_settings(root=tmp_path)

    assert settings.api.token == "from-yaml"


def test_default_applies_without_env_or_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate(monkeypatch, tmp_path)

    settings = load_settings(root=tmp_path)

    assert settings.api.token == "local-dev-token"


def test_quoted_and_export_dotenv_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    _write(
        tmp_path / ".env",
        "export APRIL_API_TOKEN=\"quoted token\"\nAPRIL_RUNTIME_TOKEN='single-quoted'\n",
    )

    settings = load_settings(root=tmp_path)

    assert settings.api.token == "quoted token"
    assert settings.runtime.token == "single-quoted"


def test_malformed_dotenv_lines_are_handled_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate(monkeypatch, tmp_path)
    _write(
        tmp_path / ".env",
        "\n".join(
            [
                "# a comment",
                "   ",
                "GARBAGE LINE WITHOUT EQUALS",
                "=novalue",
                "export ",
                "NOT_AN_APRIL_KEY=whatever",
                "bad key=value",
                "APRIL_API_TOKEN=good-token  # trailing comment",
                "APRIL_RUNTIME_PORT=9001",
            ]
        )
        + "\n",
    )

    settings = load_settings(root=tmp_path)

    assert settings.api.token == "good-token"
    assert settings.runtime.port == 9001


def test_dotenv_cannot_relocate_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _write(tmp_path / ".env", f"APRIL_HOME={elsewhere}\n")

    settings = load_settings(root=tmp_path)

    assert settings.home == tmp_path.resolve()


def test_setup_tokens_visible_to_both_child_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate(monkeypatch, tmp_path)
    write_token_env_file(
        tmp_path / ".env",
        GeneratedTokens(api_token="api-child", runtime_token="rt-child"),
    )

    # Each service (core API, runtime, CLI child) independently calls load_settings
    # under the same APRIL_HOME and must converge on the same effective tokens.
    first = load_settings(root=tmp_path)
    second = load_settings(root=tmp_path)
    assert first.api.token == "api-child"
    assert first.runtime.token == "rt-child"
    assert (second.api.token, second.runtime.token) == (
        first.api.token,
        first.runtime.token,
    )

    # The service manager hands children APRIL_HOME rather than the raw secrets,
    # so children re-read the same .env instead of receiving tokens via argv/env.
    manager = AprilServiceManager(home=tmp_path)
    child_env = manager._child_env(fake_backend=True)
    assert child_env["APRIL_HOME"] == str(tmp_path)
    assert "APRIL_API_TOKEN" not in child_env
    assert "APRIL_RUNTIME_TOKEN" not in child_env
    assert manager.settings.api.token == "api-child"
    assert manager.settings.runtime.token == "rt-child"


def test_secrets_not_exposed_by_repr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    _write(
        tmp_path / ".env",
        "APRIL_API_TOKEN=super-secret-api\nAPRIL_RUNTIME_TOKEN=super-secret-rt\n",
    )

    settings = load_settings(root=tmp_path)

    assert settings.api.token == "super-secret-api"
    assert settings.runtime.token == "super-secret-rt"
    for rendered in (repr(settings), repr(settings.api), repr(settings.runtime)):
        assert "super-secret-api" not in rendered
        assert "super-secret-rt" not in rendered


def test_generated_token_file_is_user_only_on_posix(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("POSIX file modes only")
    target = tmp_path / ".env"
    write_token_env_file(target, GeneratedTokens(api_token="a", runtime_token="b"))
    mode = stat.S_IMODE(os.stat(target).st_mode)
    assert mode == 0o600
