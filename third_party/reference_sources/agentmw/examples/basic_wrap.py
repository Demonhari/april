"""Minimal example. Requires `pip install -e .[anthropic]` and ANTHROPIC_API_KEY.

This file is illustrative — don't run it in CI.
"""

import os

import anthropic  # type: ignore[import-not-found]

from agentmw import wrap

client = wrap(anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]))

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    messages=[{"role": "user", "content": "Say hi in one word."}],
)
print(response)

trace = client.config.traces[-1]
print(f"\n[trace] monitors fired: {[r.name for r in trace.monitors.triggered]}")
print(f"[trace] compression saved: {trace.compression.ratio*100:.1f}%")
print(f"[trace] reasoning patterns recalled: {trace.recalled_patterns}")
