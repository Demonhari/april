"""Markdown persistence — the memory file IS the database.

Every other memory system hides your facts inside an opaque index (SQLite blobs,
vector stores, a server). mnem stores its state as a plain Markdown file that is
simultaneously:

  * the storage format — ``Memory.open("MEMORY.md")`` and every change lands there;
  * a human interface — add a line to teach it, delete a line to forget, edit a
    line to correct it; mnem reloads your edits as the new truth;
  * an audit log — superseded values stay nested and ~~struck through~~ under the
    current value, so the file shows *how* the memory evolved;
  * version-controllable — ``git diff`` your agent's brain, review it in a PR.

Format (mnem:v1)::

    # mnem

    - Actually I now prefer window seats <!-- t=1751234567.000000 -->
      - ~~I prefer aisle seats~~ <!-- t=1751230000.000000 superseded -->
    - I like quiet hotels near parks <!-- t=1751234570.000000 -->

Lines a human adds without a timestamp comment are treated as the newest facts,
so a hand-written update supersedes the stored value — exactly what you'd expect.
Private facts are NOT written to disk unless explicitly requested.
"""

from __future__ import annotations

import re
from pathlib import Path

_HEADER_LINES = [
    "# mnem",
    "",
    "<!-- mnem:v1 · this file IS the memory. Add a line to teach it, delete a line",
    "     to forget, edit a line to correct it: mnem reloads your changes as the",
    "     new truth. Nested ~~struck~~ lines are superseded history. -->",
    "",
]

_CURRENT_RE = re.compile(r"^- (?:\[(private)\]\s*)?(.+?)(?:\s*<!--\s*t=([0-9.eE+]+)\s*-->)?\s*$")
_HISTORY_RE = re.compile(r"^(?:\s{2,}|\t)- ~~(.+?)~~(?:\s*<!--\s*t=([0-9.eE+]+)[^>]*-->)?\s*$")


def _ts(raw):
    try:
        return float(raw) if raw else None
    except (TypeError, ValueError):
        return None


def load_into(memory, path) -> int:
    """Replay a memory file into ``memory``. Returns the number of facts loaded."""
    blocks: list[dict] = []
    current: dict | None = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        history = _HISTORY_RE.match(line)
        if history and current is not None:
            current["history"].append({"text": history.group(1).strip(), "ts": _ts(history.group(2))})
            continue
        match = _CURRENT_RE.match(line)
        if match:
            current = {
                "private": bool(match.group(1)),
                "text": match.group(2).strip(),
                "ts": _ts(match.group(3)),
                "history": [],
            }
            blocks.append(current)

    # Hand-added lines carry no timestamp: give them timestamps *after* every
    # stored one, so a human edit counts as the newest statement.
    known = [b["ts"] for b in blocks if b["ts"] is not None]
    known += [h["ts"] for b in blocks for h in b["history"] if h["ts"] is not None]
    auto = max(known, default=0.0)

    def next_auto() -> float:
        nonlocal auto
        auto += 1.0
        return auto

    loaded = 0
    for block in blocks:
        history_ids = []
        for item in block["history"]:
            idx = memory.add(item["text"], timestamp=item["ts"] or next_auto(), reinforce=False)
            if idx is not None:
                history_ids.append(idx)
                loaded += 1
        idx = memory.add(
            block["text"],
            timestamp=block["ts"] or next_auto(),
            private=block["private"],
            reinforce=True,
        )
        if idx is None:
            continue
        loaded += 1
        # Nesting in the file is an explicit supersession statement — honor it
        # even where lexical overlap alone would not have triggered it.
        for hid in history_ids:
            fact = memory._facts[hid]
            if fact.superseded_by is None:
                fact.superseded_by = idx
                memory._unindex(hid)
    return loaded


def dump_to(memory, path, *, include_private: bool = False) -> None:
    """Write ``memory`` to ``path`` in the mnem:v1 Markdown format."""
    facts = memory._facts

    # Group superseded facts under the current fact that ended their chain.
    chains: dict[int, list[int]] = {}
    for idx, fact in enumerate(facts):
        if fact.removed or fact.superseded_by is None:
            continue
        terminal = idx
        while facts[terminal].superseded_by is not None:
            terminal = facts[terminal].superseded_by
        if not facts[terminal].removed:
            chains.setdefault(terminal, []).append(idx)

    lines = list(_HEADER_LINES)
    for idx in sorted(range(len(facts)), key=lambda i: facts[i].timestamp):
        fact = facts[idx]
        if fact.removed or fact.superseded_by is not None:
            continue
        if fact.private and not include_private:
            continue
        marker = "[private] " if fact.private else ""
        lines.append(f"- {marker}{fact.text} <!-- t={fact.timestamp:.6f} -->")
        for hid in sorted(chains.get(idx, []), key=lambda i: facts[i].timestamp):
            older = facts[hid]
            if older.removed or (older.private and not include_private):
                continue
            lines.append(f"  - ~~{older.text}~~ <!-- t={older.timestamp:.6f} superseded -->")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
