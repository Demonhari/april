from __future__ import annotations

import stat

import pytest

from april_common.audit import redact
from april_common.errors import ConfigError
from april_common.settings import load_settings, reset_settings_cache
from april_common.token_setup import generate_tokens, write_token_env_file


def test_default_tokens_rejected_outside_development_and_test(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APRIL_ENV", "production")
    monkeypatch.setenv("APRIL_HOME", str(tmp_path))
    reset_settings_cache()
    with pytest.raises(ConfigError, match="Known development tokens"):
        load_settings(root=tmp_path)
    reset_settings_cache()


def test_secure_token_generation_and_env_file_permissions(tmp_path) -> None:
    tokens = generate_tokens()
    assert tokens.api_token != tokens.runtime_token
    assert len(tokens.api_token) >= 32
    assert len(tokens.runtime_token) >= 32

    env_file = tmp_path / ".env"
    write_token_env_file(env_file, tokens)
    text = env_file.read_text(encoding="utf-8")
    assert f"APRIL_API_TOKEN={tokens.api_token}" in text
    assert f"APRIL_RUNTIME_TOKEN={tokens.runtime_token}" in text
    mode = stat.S_IMODE(env_file.stat().st_mode)
    assert mode == 0o600


def test_authorization_header_sanitization() -> None:
    redacted = redact({"headers": {"Authorization": "Bearer secret-token"}})
    assert redacted["headers"]["Authorization"] == "[REDACTED]"
