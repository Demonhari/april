"""Test the Ollama-backed LLM judge monitor.

- Always tests the "Ollama unreachable" path with a deliberately-wrong host.
- Tests the parsing path with a mocked urlopen.
- Skips the live Ollama test unless OLLAMA_HOST is reachable.
"""

from __future__ import annotations

import io
import json
from unittest.mock import patch

from agentmw.core.llm_judge import llm_judge_monitor


def test_judge_skips_when_ollama_unreachable():
    """If Ollama is not running, the monitor must NOT fire (graceful skip)."""
    messages = [
        {"role": "user", "content": "do a thing"},
        {"role": "assistant", "content": [{"type": "text", "text": "Actually let me reconsider."}]},
    ]
    result = llm_judge_monitor(messages, host="http://127.0.0.1:1")  # bogus port
    assert not result.triggered
    assert "unreachable" in result.reason.lower() or result.reason == ""


def test_judge_parses_problem_response():
    """When Ollama replies with a problem, the monitor must fire."""
    messages = [
        {"role": "user", "content": "Implement the function"},
        {"role": "assistant", "content": [{"type": "text", "text": "I previously said X but now Y."}]},
    ]
    fake_judge_response = json.dumps({
        "response": '{"problem": "contradiction", "reason": "Reversed earlier claim about X."}'
    }).encode()

    def fake_urlopen(req, *args, **kwargs):
        url = req.full_url if hasattr(req, "full_url") else req
        body = b'{"models":[]}' if str(url).endswith("/api/tags") else fake_judge_response
        return io.BytesIO(body)

    with patch("agentmw.core.llm_judge.urllib.request.urlopen", side_effect=fake_urlopen):
        result = llm_judge_monitor(messages)

    assert result.triggered
    assert "contradiction" in result.reason
    assert "reviewer flagged" in result.correction


def test_judge_does_not_fire_when_problem_is_none():
    messages = [
        {"role": "user", "content": "Implement the function"},
        {"role": "assistant", "content": [{"type": "text", "text": "Sure, I'll do it."}]},
    ]
    fake_judge_response = json.dumps({
        "response": '{"problem": "none", "reason": ""}'
    }).encode()

    def fake_urlopen(req, *args, **kwargs):
        url = req.full_url if hasattr(req, "full_url") else req
        body = b'{"models":[]}' if str(url).endswith("/api/tags") else fake_judge_response
        return io.BytesIO(body)

    with patch("agentmw.core.llm_judge.urllib.request.urlopen", side_effect=fake_urlopen):
        result = llm_judge_monitor(messages)

    assert not result.triggered
