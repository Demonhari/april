"""Context compression: trim stale tool_results from older turns.

Strategy: keep the most recent `keep_recent` tool_result blocks intact;
older ones are replaced with a short placeholder so the model still
sees the call happened but the (often large) payload doesn't pay for
itself on every subsequent call.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

Message = dict[str, Any]


@dataclass
class CompressionStats:
    bytes_before: int
    bytes_after: int
    truncated_blocks: int

    @property
    def ratio(self) -> float:
        return 0.0 if self.bytes_before == 0 else 1.0 - (self.bytes_after / self.bytes_before)


def _approx_size(messages: list[Message]) -> int:
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            n += len(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict):
                    for v in b.values():
                        n += len(str(v))
    return n


def compress_history(
    messages: list[Message],
    keep_recent: int = 2,
    max_tool_result_chars: int = 240,
) -> tuple[list[Message], CompressionStats]:
    """Truncate old tool_result blocks while preserving structure.

    - `keep_recent`: number of most-recent tool_result blocks kept verbatim.
    - `max_tool_result_chars`: characters retained from truncated blocks (head).
    """
    out = copy.deepcopy(messages)
    bytes_before = _approx_size(out)

    tool_result_positions: list[tuple[int, int]] = []
    for i, msg in enumerate(out):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for j, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tool_result_positions.append((i, j))

    to_truncate = tool_result_positions[:-keep_recent] if keep_recent > 0 else tool_result_positions
    truncated = 0
    for i, j in to_truncate:
        block = out[i]["content"][j]
        content = block.get("content")
        if isinstance(content, str):
            if len(content) > max_tool_result_chars:
                block["content"] = content[:max_tool_result_chars] + " …[truncated by agentmw]"
                truncated += 1
        elif isinstance(content, list):
            for sub in content:
                if isinstance(sub, dict) and sub.get("type") == "text":
                    txt = sub.get("text", "")
                    if len(txt) > max_tool_result_chars:
                        sub["text"] = txt[:max_tool_result_chars] + " …[truncated by agentmw]"
                        truncated += 1

    bytes_after = _approx_size(out)
    return out, CompressionStats(
        bytes_before=bytes_before,
        bytes_after=bytes_after,
        truncated_blocks=truncated,
    )
