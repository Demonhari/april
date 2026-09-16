"""Tests for the layered config: defaults < TOML file < env vars."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

from agentmw.core.config import AgentmwConfig, default_config


def test_defaults_are_sane():
    cfg = AgentmwConfig()
    assert cfg.monitors.loop_threshold >= 2
    assert cfg.compression.keep_recent >= 0
    assert 0.0 < cfg.memory.semantic_threshold <= 1.0
    assert cfg.provider.name == "auto"
    assert cfg.pipeline.use_llm is True


def test_env_overrides_defaults():
    with patch.dict(os.environ, {
        "AGENTMW_LOOP_THRESHOLD": "7",
        "AGENTMW_PROVIDER": "openai",
        "AGENTMW_MODEL": "gpt-4o",
        "AGENTMW_USE_LLM": "false",
        "AGENTMW_SEMANTIC_THRESHOLD": "0.42",
    }, clear=False):
        cfg = AgentmwConfig.from_env_and_file(config_path="/nonexistent/path.toml")
    assert cfg.monitors.loop_threshold == 7
    assert cfg.provider.name == "openai"
    assert cfg.provider.model == "gpt-4o"
    assert cfg.pipeline.use_llm is False
    assert cfg.memory.semantic_threshold == 0.42


def test_toml_file_overrides_defaults_but_env_wins():
    toml_body = b"""
[monitors]
loop_threshold = 4

[provider]
name = "anthropic"
model = "claude-haiku-4-5-20251001"

[pipeline]
use_llm = true
"""
    with tempfile.NamedTemporaryFile("wb", suffix=".toml", delete=False) as f:
        f.write(toml_body)
        path = f.name
    try:
        with patch.dict(os.environ, {"AGENTMW_PROVIDER": "openai"}, clear=False):
            cfg = AgentmwConfig.from_env_and_file(config_path=path)
        # TOML applied
        assert cfg.monitors.loop_threshold == 4
        # env still wins on top
        assert cfg.provider.name == "openai"
        # TOML applied where env didn't override
        assert cfg.provider.model == "claude-haiku-4-5-20251001"
    finally:
        os.unlink(path)


def test_invalid_env_int_falls_back_to_default():
    with patch.dict(os.environ, {"AGENTMW_LOOP_THRESHOLD": "not-a-number"}, clear=False):
        cfg = AgentmwConfig.from_env_and_file(config_path="/nonexistent/path.toml")
    assert cfg.monitors.loop_threshold == AgentmwConfig().monitors.loop_threshold
