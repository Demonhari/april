"""Tests for record/sessions/replay (no live LLM calls)."""

from __future__ import annotations

import os
import tempfile

from agentmw.core.replay import build_counterfactual, render_counterfactual
from agentmw.core.sessions import Session, SessionStore
from agentmw.core.timeline import build_timeline


class _ScriptedProvider:
    name = "scripted"
    model = "scripted-1"
    available = True

    def __init__(self, response: str):
        self._response = response

    def judge(self, system: str, prompt: str) -> str:
        return self._response


SAMPLE = [
    {"role": "user", "content": "fix the email-drop bug"},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "grep", "input": {"q": "retry"}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "retry.py:14"}]},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "b", "name": "grep", "input": {"q": "retry"}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "b", "content": "retry.py:14"}]},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "c", "name": "grep", "input": {"q": "retry"}}]},
]


def test_session_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(directory=tmp)
        s = Session.from_messages(SAMPLE)
        assert s.task.startswith("fix the email-drop")
        store.save(s)
        loaded = store.load(s.id)
        assert loaded.id == s.id
        assert len(loaded.messages) == len(SAMPLE)


def test_session_prefix_lookup():
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(directory=tmp)
        s = Session.from_messages(SAMPLE)
        store.save(s)
        loaded = store.load(s.id[:6])
        assert loaded.id == s.id


def test_counterfactual_renders_ghost_steps():
    report = build_timeline(SAMPLE, config=None)
    assert report.first_fire_step is not None

    provider = _ScriptedProvider(
        '{"steps": ['
        '{"action": "Read", "rationale": "Open retry.py and inspect the pop() call directly."},'
        '{"action": "Edit", "rationale": "Remove the .pop(\'email\') line; obsolete PII guard."}'
        ']}'
    )
    cf = build_counterfactual(SAMPLE, report, provider)
    assert cf is not None
    assert len(cf.ghost_steps) == 2
    rendered = render_counterfactual(report, cf)
    assert "COUNTERFACTUAL" in rendered
    assert "GHOST" in rendered
    assert "Read" in rendered or "Edit" in rendered
