"""Tests for the provider abstraction. All network calls are mocked."""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest

from agentmw.core.config import ProviderConfig
from agentmw.core.providers import (
    AnthropicProvider,
    NoneProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
    ProviderError,
    select_provider,
)


def _fake_response(body: dict | bytes):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    return io.BytesIO(data)


def test_none_provider_is_not_available():
    p = NoneProvider()
    assert p.available is False
    with pytest.raises(ProviderError):
        p.judge("sys", "user")


def test_openai_judge_extracts_message_content():
    cfg = ProviderConfig(name="openai", api_key="sk-test", model="gpt-4o-mini")
    p = OpenAIProvider(cfg)
    assert p.available is True

    fake = _fake_response({"choices": [{"message": {"content": "  Hello back  "}}]})
    with patch("agentmw.core.providers.urllib.request.urlopen", return_value=fake) as mock:
        out = p.judge("be brief", "hi")
    assert out == "Hello back"
    sent_req = mock.call_args.args[0]
    assert "Bearer sk-test" in dict(sent_req.headers).get("Authorization", "")


def test_anthropic_judge_extracts_text_block():
    cfg = ProviderConfig(name="anthropic", api_key="ak-test")
    p = AnthropicProvider(cfg)
    fake = _fake_response({"content": [{"type": "text", "text": "ok!"}]})
    with patch("agentmw.core.providers.urllib.request.urlopen", return_value=fake):
        out = p.judge("be brief", "ping")
    assert out == "ok!"


def test_openrouter_inherits_openai_shape():
    cfg = ProviderConfig(name="openrouter", api_key="or-test")
    p = OpenRouterProvider(cfg)
    assert "openrouter.ai" in p.base_url
    fake = _fake_response({"choices": [{"message": {"content": "router reply"}}]})
    with patch("agentmw.core.providers.urllib.request.urlopen", return_value=fake):
        out = p.judge("sys", "prompt")
    assert out == "router reply"


def test_ollama_judge_extracts_response_field():
    cfg = ProviderConfig(name="ollama", base_url="http://localhost:11434", model="qwen3")
    p = OllamaProvider(cfg)

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        if url.endswith("/api/tags"):
            return _fake_response(b'{"models":[]}')
        return _fake_response({"response": "ollama reply"})

    with patch("agentmw.core.providers.urllib.request.urlopen", side_effect=fake_urlopen):
        assert p.available is True
        out = p.judge("sys", "prompt")
    assert out == "ollama reply"


def test_select_provider_explicit_name_wins():
    cfg = ProviderConfig(name="openai", api_key="sk-x")
    p = select_provider(cfg)
    assert p.name == "openai"


def test_select_provider_auto_falls_back_to_none_when_nothing_available():
    cfg = ProviderConfig(name="auto", api_key=None)
    # Force Ollama unreachable by aiming at a bogus host
    cfg.base_url = "http://127.0.0.1:1"
    p = select_provider(cfg)
    assert p.name == "none"


def test_retry_eventually_raises_provider_error():
    cfg = ProviderConfig(name="openai", api_key="sk-x", max_retries=1, timeout_seconds=0.1)
    p = OpenAIProvider(cfg)
    import urllib.error
    with patch(
        "agentmw.core.providers.urllib.request.urlopen",
        side_effect=urllib.error.URLError("boom"),
    ):
        with pytest.raises(ProviderError):
            p.judge("s", "p")
