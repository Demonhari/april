"""Tests for the orchestrated monitor pipeline (LLM primary + heuristics fallback)."""

from __future__ import annotations

from agentmw.core.config import AgentmwConfig
from agentmw.core.monitors import run_monitors
from agentmw.core.providers import NoneProvider, ProviderError


class _FakeProvider:
    name = "fake"
    model = "fake-1"
    available = True

    def __init__(self, response: str):
        self._response = response

    def judge(self, system: str, prompt: str) -> str:
        return self._response


class _DownProvider:
    name = "fake-down"
    model = "fake-1"
    available = True  # appears available but call raises

    def judge(self, system: str, prompt: str) -> str:
        raise ProviderError("simulated network failure")


LOOPING_TRACE = [
    {"role": "user", "content": "fix it"},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "grep", "input": {"q": "x"}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "..."}]},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "b", "name": "grep", "input": {"q": "x"}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "b", "content": "..."}]},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "c", "name": "grep", "input": {"q": "x"}}]},
]


def _cfg_llm_only() -> AgentmwConfig:
    cfg = AgentmwConfig()
    cfg.pipeline.use_llm = True
    cfg.pipeline.heuristics_prefilter = False
    cfg.pipeline.use_heuristics_fallback = True
    cfg.provider.name = "openai"
    cfg.provider.api_key = "sk-test"
    return cfg


def test_llm_primary_replaces_heuristics_when_provider_works():
    cfg = _cfg_llm_only()
    fake = _FakeProvider(
        '{"issues": [{"type": "loop", "reason": "grep called 3x with q=x"}]}'
    )
    report = run_monitors(LOOPING_TRACE, config=cfg, provider=fake)
    assert report.used_llm is True
    assert report.provider_name == "fake"
    names = [r.name for r in report.triggered]
    assert "llm:loop" in names


def test_heuristics_fallback_when_provider_fails():
    cfg = _cfg_llm_only()
    down = _DownProvider()
    report = run_monitors(LOOPING_TRACE, config=cfg, provider=down)
    assert report.used_llm is False
    assert report.used_heuristics is True
    names = {r.name for r in report.triggered}
    assert "loop" in names or "redundant_tool_call" in names


def test_pure_heuristics_when_llm_disabled():
    cfg = AgentmwConfig()
    cfg.pipeline.use_llm = False
    report = run_monitors(LOOPING_TRACE, config=cfg, provider=NoneProvider())
    assert report.used_llm is False
    assert report.used_heuristics is True


def test_dedup_prefers_llm_verdict():
    cfg = AgentmwConfig()
    cfg.pipeline.use_llm = True
    cfg.pipeline.heuristics_prefilter = True
    fake = _FakeProvider(
        '{"issues": [{"type": "loop", "reason": "the LLM verdict wins"}]}'
    )
    report = run_monitors(LOOPING_TRACE, config=cfg, provider=fake)
    triggered_loops = [r for r in report.triggered if "loop" in r.name]
    assert len(triggered_loops) == 1
    assert triggered_loops[0].name == "llm:loop"
    assert "LLM verdict wins" in triggered_loops[0].reason
