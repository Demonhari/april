"""Behavioural tests for mnem. Run with `pytest`, or directly: `python tests/test_core.py`."""

from mnem import Memory


def test_supersedes_stale_value():
    m = Memory()
    m.add("I prefer aisle seats on flights", timestamp=1)
    m.add("Actually I now prefer window seats on flights", timestamp=2)
    state = m.state()
    assert any("window" in s for s in state)
    assert not any("aisle" in s for s in state)  # stale value dropped
    assert m.recall("which seat do I like?")[0].lower().find("window") >= 0


def test_unrelated_facts_coexist():
    m = Memory()
    m.add("I prefer window seats", timestamp=1)
    m.add("I like quiet hotels near parks", timestamp=2)
    state = m.state()
    assert any("window" in s for s in state)
    assert any("hotels" in s for s in state)
    assert len(m) == 2


def test_private_facts_never_surface():
    m = Memory()
    m.add("My favourite city is Lisbon", timestamp=1)
    m.add("My passport number is 123-45-6789", timestamp=2, private=True)
    assert not any("passport" in s.lower() for s in m.state())
    assert not any("passport" in s.lower() for s in m.recall("what is my passport number"))


def test_forget_removes_fact():
    m = Memory()
    m.add("I live in Rome", timestamp=1)
    assert m.forget("Rome") == 1
    assert m.state() == []


def test_reinforcement_counts_updates():
    m = Memory()
    m.add("I prefer tea", timestamp=1)
    m.add("I still prefer tea over coffee", timestamp=2)  # same topic -> supersede + reinforce
    current = [f for f in m.recall_facts("drink") if "tea" in f.text]
    assert current and current[0].reinforce >= 2


def test_semantic_mode_uses_embeddings():
    # A tiny deterministic "embedding": bag-of-chars over a fixed alphabet.
    alphabet = "abcdefghijklmnopqrstuvwxyz "

    def embed(text):
        vec = [0.0] * len(alphabet)
        for ch in text.lower():
            i = alphabet.find(ch)
            if i >= 0:
                vec[i] += 1.0
        return vec

    m = Memory(embed=embed, supersede_threshold=0.9)
    m.add("hotel preference: quiet rooms", timestamp=1)
    m.add("hotel preference: quiet rooms away from elevators", timestamp=2)
    # near-duplicate topic should supersede under the semantic model
    assert len(m) == 1
    assert "elevators" in m.state()[0]


def test_builtin_semantic_mode_matches_morphology():
    # Memory(semantic=True) uses the built-in zero-dependency HashingEmbedder,
    # whose character n-grams match morphological variants that exact-token
    # overlap would miss ("run" vs "running").
    m = Memory(semantic=True)
    m.add("I go running every morning", timestamp=1)
    m.add("I sold my old bicycle", timestamp=1)  # same time: isolate the semantic signal
    hits = m.recall("how much do you run?", k=1)
    assert hits and "running" in hits[0]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
