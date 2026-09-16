"""DOES agentmw ACTUALLY CHANGE THE NEXT DECISION IN REAL TIME?

The honest test: take a looping trace, ask a real LLM "what would you do
next?" twice — once raw, once with agentmw's system-note correction
appended — and diff the answers.

If the corrected answer differs from the baseline in a way that breaks
the loop, then yes, agentmw nudges the agent in real time.

This script uses Nemotron via Ollama Cloud for both the baseline and the
corrected call. It does NOT mock anything.
"""

from __future__ import annotations

import os
import sys

from agentmw.core.compression import compress_history
from agentmw.core.config import default_config
from agentmw.core.memory import ReasoningLibrary
from agentmw.core.monitors import run_monitors
from agentmw.core.providers import ProviderError, select_provider


# A real looping trace, distilled from realistic_trace.json.
LOOPING_TRACE = [
    {"role": "user", "content": "Fix bug: user-creation flow drops the email field when retried."},
    {"role": "assistant", "content": [
        {"type": "text", "text": "I'll search for retry logic in the auth path."},
        {"type": "tool_use", "id": "t1", "name": "Grep", "input": {"pattern": "retry", "path": "src/auth/"}},
    ]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1",
         "content": "src/auth/retry.py:14: def retry_user_creation(payload, attempt=0):"},
    ]},
    {"role": "assistant", "content": [
        {"type": "text", "text": "Let me read retry.py."},
        {"type": "tool_use", "id": "t2", "name": "Read", "input": {"path": "src/auth/retry.py"}},
    ]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t2",
         "content": "def retry_user_creation(payload, attempt=0):\n    payload.pop('email', None)  # PII strip\n    return create_user(payload)"},
    ]},
    {"role": "assistant", "content": [
        {"type": "text", "text": "Actually, let me reconsider. Maybe the bug is upstream."},
        {"type": "tool_use", "id": "t3", "name": "Grep", "input": {"pattern": "retry", "path": "src/auth/"}},
    ]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t3",
         "content": "src/auth/retry.py:14: def retry_user_creation(payload, attempt=0):"},
    ]},
]


AGENT_SYSTEM = (
    "You are a coding agent debugging a Python repository. Decide your "
    "single next action. Respond in ONE sentence, e.g. \"I will Grep for X\" "
    "or \"I will Read path Y\" or \"I will Edit file Z to remove line N\". "
    "Be decisive — pick exactly one next step."
)


def _format_trace(messages: list) -> str:
    """Render trace as readable text the way an agent would receive it."""
    lines: list[str] = []
    for i, m in enumerate(messages, start=1):
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, str):
            lines.append(f"#{i} [{role}] {content}")
        elif isinstance(content, list):
            parts = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                t = b.get("type")
                if t == "text":
                    parts.append(b.get("text", ""))
                elif t == "tool_use":
                    inp = b.get("input", {})
                    parts.append(f"[CALLED {b.get('name')}({inp})]")
                elif t == "tool_result":
                    c = b.get("content", "")
                    parts.append(f"[TOOL_RESULT]\n{c}")
            lines.append(f"#{i} [{role}] " + " ".join(parts))
    return "\n".join(lines)


def ask_agent_baseline(provider, messages: list) -> str:
    """No agentmw intervention — agent sees the raw trace."""
    prompt = (
        "Here is the trace so far:\n\n"
        + _format_trace(messages)
        + "\n\nWhat is your next action?"
    )
    return provider.judge(AGENT_SYSTEM, prompt)


def ask_agent_with_agentmw(provider, messages: list, cfg) -> tuple[str, str]:
    """With agentmw: build the system note the way wrap() would, prepend it."""
    report = run_monitors(messages, config=cfg, provider=provider)
    correction = report.correction_text()
    system_note = ("[agentmw] Mid-run corrections:\n" + correction) if correction else ""

    augmented_system = AGENT_SYSTEM
    if system_note:
        augmented_system = AGENT_SYSTEM + "\n\n" + system_note

    prompt = (
        "Here is the trace so far:\n\n"
        + _format_trace(messages)
        + "\n\nWhat is your next action?"
    )
    return provider.judge(augmented_system, prompt), system_note


def main() -> int:
    os.environ.setdefault("AGENTMW_PROVIDER", "ollama")
    os.environ.setdefault("AGENTMW_MODEL", "nemotron-3-super:cloud")
    cfg = default_config()
    provider = select_provider(cfg.provider)

    if not provider.available:
        print(f"provider {provider.name} unavailable — aborting", file=sys.stderr)
        return 2

    print("=" * 76)
    print("REAL-TIME PROOF: does agentmw's mid-run correction change the next decision?")
    print(f"  provider: {provider.name} ({provider.model})")
    print("=" * 76)

    print("\n-- BASELINE (agent sees raw trace, no agentmw)")
    try:
        baseline = ask_agent_baseline(provider, LOOPING_TRACE)
    except ProviderError as e:
        print(f"provider error: {e}")
        return 3
    print(f"  next action: {baseline.strip()[:400]}")

    print("\n-- WITH AGENTMW (same trace + injected mid-run correction)")
    try:
        corrected, note = ask_agent_with_agentmw(provider, LOOPING_TRACE, cfg)
    except ProviderError as e:
        print(f"provider error: {e}")
        return 3
    print("  injected system note:")
    for line in note.splitlines():
        print("    " + line)
    print(f"\n  next action: {corrected.strip()[:400]}")

    print("\n" + "=" * 76)
    if baseline.strip() == corrected.strip():
        print("⚠  identical decisions — agentmw did NOT change the next step")
    else:
        print("✓  decisions differ — agentmw nudged the agent away from the looping path")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
