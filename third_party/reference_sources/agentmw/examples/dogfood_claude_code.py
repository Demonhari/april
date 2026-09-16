"""Dogfood test: feed agentmw a realistic Claude-Code-style failure trace
and see what it would have done.

The trace below mirrors a real failure mode I (Claude) exhibit when:
- I grep for the same symbol multiple times with slight variations
- I re-read a file I already read
- I second-guess my own analysis in plain text
- Tool outputs from early in the run are still bloating context
"""

from agentmw import wrap
from agentmw.wrap import WrapConfig


class _MockMessages:
    def create(self, **kwargs):
        return {"id": "mock", "messages_received": len(kwargs.get("messages", []))}


class _MockClient:
    def __init__(self):
        self.messages = _MockMessages()


CLAUDE_CODE_TRACE = [
    {"role": "user", "content": "Find why the user-creation flow drops the email field on retry, and fix it."},

    {"role": "assistant", "content": [
        {"type": "text", "text": "I'll start by searching for the retry logic."},
        {"type": "tool_use", "id": "t1", "name": "Grep", "input": {"pattern": "retry", "path": "src/"}},
    ]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1",
         "content": "src/auth/retry.py:14: def retry_user_creation(payload):\n" + ("[200 more matching lines]" * 60)},
    ]},

    {"role": "assistant", "content": [
        {"type": "text", "text": "Let me look at retry_user_creation specifically."},
        {"type": "tool_use", "id": "t2", "name": "Read", "input": {"path": "src/auth/retry.py"}},
    ]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t2",
         "content": "def retry_user_creation(payload):\n    payload.pop('email', None)\n    return create_user(payload)\n" + ("[file body padding]" * 50)},
    ]},

    {"role": "assistant", "content": [
        {"type": "text", "text": "Actually, let me reconsider. Maybe the issue is upstream."},
        {"type": "tool_use", "id": "t3", "name": "Grep", "input": {"pattern": "retry", "path": "src/"}},
    ]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t3",
         "content": "src/auth/retry.py:14: def retry_user_creation(payload):\n" + ("[200 more matching lines]" * 60)},
    ]},

    {"role": "assistant", "content": [
        {"type": "text", "text": "Wait, no, that's wrong. Let me re-grep."},
        {"type": "tool_use", "id": "t4", "name": "Grep", "input": {"pattern": "retry", "path": "src/"}},
    ]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t4",
         "content": "src/auth/retry.py:14: def retry_user_creation(payload):\n" + ("[200 more matching lines]" * 60)},
    ]},

    {"role": "assistant", "content": [
        {"type": "text", "text": "On second thought, I should just read retry.py again."},
        {"type": "tool_use", "id": "t5", "name": "Read", "input": {"path": "src/auth/retry.py"}},
    ]},
]


def main():
    cfg = WrapConfig()
    client = wrap(_MockClient(), config=cfg)

    client.memory.save(
        "Find why the user-creation flow drops the email field on retry, and fix it.",
        "When investigating field-loss bugs, look for `.pop(` or `del` on the field name first — that's been the root cause 3/3 times.",
        outcome="success",
    )

    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="You are a code-fixing agent.",
        messages=CLAUDE_CODE_TRACE,
    )

    trace = cfg.traces[-1]

    print("=" * 70)
    print("DOGFOOD: Claude-Code-style trace passed through agentmw")
    print("=" * 70)
    print(f"\nMonitors that fired: {len(trace.monitors.triggered)}")
    for r in trace.monitors.triggered:
        print(f"  ✗ [{r.name}] {r.reason}")
    print(f"\nCompression:")
    print(f"  bytes before: {trace.compression.bytes_before:>10,}")
    print(f"  bytes after:  {trace.compression.bytes_after:>10,}")
    print(f"  saved:        {trace.compression.ratio*100:>10.1f}%")
    print(f"  blocks truncated: {trace.compression.truncated_blocks}")
    print(f"\nReasoning patterns recalled: {trace.recalled_patterns}")
    print(f"\nSystem note injected into next call:")
    print("-" * 70)
    print(trace.system_note)
    print("-" * 70)


if __name__ == "__main__":
    main()
