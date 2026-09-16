"""End-to-end production demo:

- Mock Anthropic-style client (we don't want to spend tokens for the inner LLM)
- Real provider for monitors + auto-extract (Nemotron via Ollama cloud)
- Auto-record + auto-extract + telemetry + circuit breaker all ON

Run with:
    AGENTMW_PROVIDER=ollama AGENTMW_MODEL=nemotron-3-super:cloud \\
        python examples/production_e2e.py
"""

from __future__ import annotations

import os
import tempfile
import time

from agentmw import wrap
from agentmw.core.config import default_config
from agentmw.core.memory import ReasoningLibrary
from agentmw.core.sessions import SessionStore
from agentmw.core.telemetry import Telemetry


class MockMessages:
    def create(self, **kwargs):
        return {
            "id": "msg_mock",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Done. I fixed the bug by removing the .pop('email') call in retry.py."}],
        }


class MockClient:
    def __init__(self):
        self.messages = MockMessages()


def main() -> None:
    # Isolate state to a temp dir so the demo is reproducible.
    tmp = tempfile.mkdtemp(prefix="agentmw-e2e-")
    os.environ["AGENTMW_HOME"] = tmp
    os.environ.setdefault("AGENTMW_PROVIDER", "ollama")
    os.environ.setdefault("AGENTMW_MODEL", "nemotron-3-super:cloud")

    from agentmw.core.telemetry import reset_global_telemetry
    reset_global_telemetry()  # AGENTMW_HOME just changed; rebuild singleton

    cfg = default_config()
    cfg.extractor.background = False  # synchronous so we can observe results here

    print("=" * 70)
    print("agentmw — production end-to-end")
    print("=" * 70)
    print(f"home:       {tmp}")
    print(f"provider:   {cfg.provider.name} / {cfg.provider.model}")
    print(f"recorder:   {cfg.recorder.enabled}  extractor: {cfg.extractor.enabled}")
    print(f"breaker:    {cfg.breaker.enabled}   telemetry: {cfg.telemetry.enabled}")
    print()

    lib = ReasoningLibrary(db_path=cfg.memory.db_path)
    client = wrap(MockClient(), agentmw_config=cfg, memory=lib)
    print(f"wrapped client session id: {client.session_id}")
    print(f"selected provider: {client.provider.name} (available={client.provider.available})")
    print()

    # Simulate a session that loops Grep three times (a real-world failure mode).
    messages = [
        {"role": "user", "content": "Fix the user-creation flow that drops email on retry."},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Grep", "input": {"q": "retry"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "retry.py:14"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "Grep", "input": {"q": "retry"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "retry.py:14"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t3", "name": "Grep", "input": {"q": "retry"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t3", "content": "retry.py:14"}]},
    ]

    t0 = time.time()
    client.messages.create(
        model="claude-mock",
        max_tokens=512,
        system="You are a code-fixing agent.",
        messages=messages,
    )
    elapsed = time.time() - t0

    trace = client.config.traces[-1]
    print(f"call elapsed: {elapsed:.2f}s")
    print(f"monitors fired ({len(trace.monitors.triggered)}):")
    for r in trace.monitors.triggered:
        print(f"  ⚠  {r.name}: {r.reason}")
    print(f"compression: {trace.compression.bytes_before} → {trace.compression.bytes_after} "
          f"({trace.compression.ratio*100:.1f}% saved)")
    print(f"recalled patterns: {trace.recalled_patterns}")
    print()

    store = SessionStore()
    sessions = store.list()
    print(f"sessions on disk: {len(sessions)}")
    for s in sessions[:3]:
        print(f"  {s.id}  {len(s.messages)} msg  {s.task[:50]}")
    print()

    print(f"reasoning library now has {lib.count()} patterns total")
    print()

    from pathlib import Path
    from agentmw.core.telemetry import global_telemetry
    global_telemetry().save()  # ensure flushed before reading
    tel = Telemetry.load(Path(tmp) / "telemetry.json")
    print("telemetry:")
    import json
    print(json.dumps(tel.to_dict(), indent=2))


if __name__ == "__main__":
    main()
