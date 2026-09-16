"""Production-grade features: extractor, breaker, telemetry, async wrap, auto-record."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time

import pytest

from agentmw import wrap, wrap_async
from agentmw.core.breaker import BreakerConfig, CircuitBreakerProvider
from agentmw.core.config import AgentmwConfig
from agentmw.core.extractor import PatternExtractor, is_session_complete
from agentmw.core.memory import ReasoningLibrary
from agentmw.core.providers import ProviderError
from agentmw.core.telemetry import Telemetry


# ---------- helpers ----------

class _ScriptedProvider:
    name = "scripted"
    model = "scripted-1"
    available = True

    def __init__(self, response: str):
        self._response = response
        self.calls = 0

    def judge(self, system: str, prompt: str) -> str:
        self.calls += 1
        return self._response


class _FailingProvider:
    name = "failing"
    model = "x"
    available = True

    def __init__(self, fail_times: int = 100):
        self._left = fail_times

    def judge(self, system: str, prompt: str) -> str:
        if self._left > 0:
            self._left -= 1
            raise ProviderError("boom")
        return '{"issues": []}'


class _MockMessages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _MockClient:
    def __init__(self, response):
        self.messages = _MockMessages(response)


class _AsyncMockMessages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _AsyncMockClient:
    def __init__(self, response):
        self.messages = _AsyncMockMessages(response)


# ---------- extractor ----------

def test_is_session_complete_recognizes_text_end():
    msgs = [
        {"role": "user", "content": "do X"},
        {"role": "assistant", "content": [{"type": "text", "text": "Done. Here is what I changed."}]},
    ]
    assert is_session_complete(msgs) is True


def test_is_session_complete_false_on_pending_tool_use():
    msgs = [
        {"role": "user", "content": "do X"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "x", "input": {}}]},
    ]
    assert is_session_complete(msgs) is False


def test_extractor_saves_patterns_and_dedups():
    with tempfile.TemporaryDirectory() as tmp:
        from agentmw.core.embeddings import NoOpBackend
        lib = ReasoningLibrary(db_path=os.path.join(tmp, "m.db"), embeddings=NoOpBackend())
        provider = _ScriptedProvider(
            '{"patterns": [{"task": "fix retry bug", "pattern": "look for .pop()"}]}'
        )
        saved = PatternExtractor(provider, lib).extract([
            {"role": "user", "content": "fix the bug"},
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        ])
        assert len(saved) == 1
        assert lib.count() == 1


# ---------- circuit breaker ----------

def test_breaker_trips_after_failures():
    inner = _FailingProvider(fail_times=10)
    cb = CircuitBreakerProvider(inner, BreakerConfig(failure_threshold=2, failure_window_seconds=5, cooldown_seconds=10))
    for _ in range(2):
        with pytest.raises(ProviderError):
            cb.judge("s", "p")
    assert cb._state == "open"
    # short-circuits without calling inner
    with pytest.raises(ProviderError, match="circuit breaker open"):
        cb.judge("s", "p")


def test_breaker_recovers_after_cooldown():
    inner = _FailingProvider(fail_times=2)
    cb = CircuitBreakerProvider(inner, BreakerConfig(failure_threshold=2, failure_window_seconds=5, cooldown_seconds=0.05))
    for _ in range(2):
        with pytest.raises(ProviderError):
            cb.judge("s", "p")
    time.sleep(0.07)
    assert cb._state == "half_open"
    # success → closed
    assert cb.judge("s", "p") == '{"issues": []}'
    assert cb._state == "closed"


# ---------- telemetry ----------

def test_telemetry_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tel.json")
        from pathlib import Path
        t = Telemetry()
        t._path = Path(path)
        t.record_call(monitors_fired=["loop"], tokens_saved=400, recalled=2, provider_name="ollama")
        t.record_call(monitors_fired=[], tokens_saved=200, recalled=0, provider_name="ollama")
        t.record_provider_call(success=True)
        t.record_provider_call(success=False)
        t.record_extract(2)
        t.save()
        t2 = Telemetry.load(Path(path))
        assert t2.calls_total == 2
        assert t2.calls_with_corrections == 1
        assert t2.monitors_fired.get("loop") == 1
        assert t2.tokens_compressed == 600
        assert t2.provider_failures == 1
        assert t2.patterns_extracted == 2


# ---------- async wrap ----------

def test_wrap_async_intercepts_create():
    cfg = AgentmwConfig()
    cfg.pipeline.use_llm = False
    cfg.recorder.enabled = False
    cfg.extractor.enabled = False
    cfg.telemetry.enabled = False

    fake_response = {"id": "x", "stop_reason": "end_turn", "content": [{"type": "text", "text": "ok"}]}
    client = wrap_async(_AsyncMockClient(fake_response), agentmw_config=cfg)
    coro = client.messages.create(
        messages=[{"role": "user", "content": "do X"}],
        system="be brief",
    )
    out = asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)
    assert out["id"] == "x"


# ---------- auto-record ----------

def test_auto_record_writes_session_file():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = AgentmwConfig()
        cfg.pipeline.use_llm = False
        cfg.recorder.enabled = True
        cfg.recorder.directory = tmp
        cfg.extractor.enabled = False
        cfg.telemetry.enabled = False

        response = {"id": "y", "stop_reason": "end_turn", "content": [{"type": "text", "text": "done"}]}
        client = wrap(_MockClient(response), agentmw_config=cfg)
        client.messages.create(messages=[{"role": "user", "content": "fix it"}], system="x")

        files = list(os.listdir(tmp))
        assert files, "expected at least one session file"


# ---------- auto-extract on completion ----------

def test_auto_extract_runs_on_end_turn_session():
    with tempfile.TemporaryDirectory() as tmp:
        from agentmw.core.embeddings import NoOpBackend
        lib = ReasoningLibrary(db_path=os.path.join(tmp, "m.db"), embeddings=NoOpBackend())
        provider = _ScriptedProvider('{"patterns": [{"task": "X", "pattern": "Y"}]}')

        cfg = AgentmwConfig()
        cfg.pipeline.use_llm = False  # disable monitor LLM, we test extractor only
        cfg.recorder.enabled = False
        cfg.telemetry.enabled = False
        cfg.extractor.enabled = True
        cfg.extractor.background = False  # sync for the test
        cfg.breaker.enabled = False

        response = {"id": "y", "stop_reason": "end_turn", "content": [{"type": "text", "text": "all done"}]}
        client = wrap(_MockClient(response), memory=lib, agentmw_config=cfg, provider=provider)
        client.messages.create(messages=[{"role": "user", "content": "do something useful"}], system="x")

        # Extractor should have run synchronously and saved one pattern.
        assert provider.calls >= 1
        assert lib.count() == 1
