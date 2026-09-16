"""Smoke tests — no network, no API key required."""

import os
import tempfile

from agentmw import wrap
from agentmw.core.compression import compress_history
from agentmw.core.memory import ReasoningLibrary
from agentmw.core.monitors import run_monitors
from agentmw.wrap import WrapConfig


class _MockMessages:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return {"id": "mock", "ok": True}


class _MockClient:
    def __init__(self):
        self.messages = _MockMessages()


def _looping_messages():
    return [
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "search", "input": {"q": "foo"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "x" * 1000}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "b", "name": "search", "input": {"q": "foo"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "b", "content": "x" * 1000}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "c", "name": "search", "input": {"q": "foo"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c", "content": "x" * 1000}]},
    ]


def test_loop_monitor_fires():
    report = run_monitors(_looping_messages())
    fired = [r.name for r in report.triggered]
    assert "loop" in fired
    assert "redundant_tool_call" in fired


def test_compression_shrinks_old_tool_results():
    msgs = _looping_messages()
    compressed, stats = compress_history(msgs, keep_recent=1, max_tool_result_chars=50)
    assert stats.bytes_after < stats.bytes_before
    assert stats.truncated_blocks >= 1


def test_memory_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "m.db")
        lib = ReasoningLibrary(db_path=db)
        assert lib.count() == 0
        lib.save("fix the auth bug", "check session token validation first", outcome="success")
        recalled = lib.recall("auth bug fix")
        assert len(recalled) >= 1
        assert "session token" in recalled[0].pattern_text
        lib.close()


def test_wrap_injects_system_note_and_calls_inner():
    msgs = _looping_messages()
    with tempfile.TemporaryDirectory() as tmp:
        lib = ReasoningLibrary(db_path=os.path.join(tmp, "m.db"))
        lib.save("do the thing", "split the problem first", outcome="success")
        cfg = WrapConfig()
        mock = _MockClient()
        client = wrap(mock, memory=lib, config=cfg)
        out = client.messages.create(messages=msgs, system="You are a helpful agent.")
        assert out["ok"] is True
        forwarded_system = mock.messages.last_kwargs["system"]
        assert "agentmw" in forwarded_system
        assert "loop" in forwarded_system or "Mid-run corrections" in forwarded_system
        assert len(cfg.traces) == 1
        trace = cfg.traces[0]
        assert trace.compression.bytes_after < trace.compression.bytes_before
        assert any(r.triggered for r in trace.monitors.results)
        lib.close()
