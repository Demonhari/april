"""Tests for the Markdown store. Run with `pytest`, or: `python tests/test_store.py`."""

import os
import tempfile
from pathlib import Path

from mnem import Memory


def _path(name):
    return os.path.join(tempfile.mkdtemp(prefix="mnem-store-"), name)


def test_roundtrip_preserves_state_and_history():
    path = _path("memory.md")
    m = Memory.open(path)
    m.add("I prefer aisle seats on flights", timestamp=1)
    m.add("Actually I now prefer window seats on flights", timestamp=2)
    m.add("I like quiet hotels near parks", timestamp=3)

    text = Path(path).read_text()
    assert "~~I prefer aisle seats on flights~~" in text  # history kept, struck through

    m2 = Memory.open(path)
    assert sorted(m2.state()) == sorted(m.state())
    assert not any("aisle" in s for s in m2.state())  # stale value stays superseded


def test_human_edit_becomes_the_new_truth():
    path = _path("memory.md")
    m = Memory.open(path)
    m.add("I live in Rome", timestamp=1)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("- I now live in Milan\n")  # a human teaches the memory

    m2 = Memory.open(path)
    state = m2.state()
    assert any("Milan" in s for s in state)
    assert not any("Rome" in s for s in state)  # hand-written update supersedes


def test_deleting_a_line_forgets():
    path = _path("memory.md")
    m = Memory.open(path)
    m.add("My favourite colour is green", timestamp=1)
    m.add("I play tennis on Sundays", timestamp=2)
    kept = [line for line in Path(path).read_text().splitlines() if "tennis" not in line]
    Path(path).write_text("\n".join(kept) + "\n")

    m2 = Memory.open(path)
    assert not any("tennis" in s for s in m2.state())
    assert any("green" in s for s in m2.state())


def test_private_facts_stay_out_of_the_file():
    path = _path("memory.md")
    m = Memory.open(path)
    m.add("My passport number is 123-45-6789", private=True)
    assert "passport" not in Path(path).read_text()


def test_prompt_respects_budget():
    m = Memory()
    for i in range(50):
        m.add(f"distinct fact number {i} about topic{i}", timestamp=i, reinforce=False)
    block = m.prompt(budget=60)
    assert block.startswith("Current facts")
    assert 0 < block.count("\n- ") < 15  # truncated well below the 50 stored facts
    assert m.prompt(budget=10_000).count("\n- ") == 50


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
