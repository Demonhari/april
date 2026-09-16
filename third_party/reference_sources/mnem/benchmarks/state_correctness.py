"""State-correctness benchmark — the capability retrieval benchmarks under-measure.

Standard memory benchmarks ask "was the relevant text retrieved?". They rarely
penalise returning a *stale* value alongside the current one. But that is exactly
where agents fail: a user updates a fact, and a similarity search hands the model
both the old and the new value.

This benchmark scores that directly. Each case states a fact, then updates it, and
asks for the value. A memory is judged on three things:

    current@1   — is the up-to-date value ranked first?
    no-stale    — does the stale value stay OUT of the top-k? (higher is better)
    clean       — current is present AND stale is absent (the only real success)

We compare mnem against two honest baselines with the same interface: a lexical
retriever (relevance only, like BM25/vector) and a recency memory. The scenarios
are ordinary restatements — the kind every real conversation contains.
"""

from __future__ import annotations

import math
import re
import sys

sys.path.insert(0, ".")
from mnem import Memory  # noqa: E402

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(text):
    return _TOKEN.findall(text.lower())


class LexicalRetriever:
    """Relevance-only retriever (no notion of state) — a stand-in for BM25/vector."""

    def __init__(self):
        self._facts = []
        self._df = {}
        self._n = 0

    def add(self, text, **_):
        self._facts.append(text)
        self._n += 1
        for t in set(_tok(text)):
            self._df[t] = self._df.get(t, 0) + 1

    def _idf(self, t):
        return math.log((1 + self._n) / (1 + self._df.get(t, 0))) + 1.0

    def recall(self, query, k=5):
        q = set(_tok(query))
        scored = []
        for text in self._facts:
            shared = set(_tok(text)) & q
            score = sum(self._idf(t) for t in shared)
            scored.append((score, text))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for s, t in scored[:k]]


class RecencyMemory:
    """Returns the most recent statements."""

    def __init__(self):
        self._facts = []

    def add(self, text, **_):
        self._facts.append(text)

    def recall(self, query, k=5):
        return list(reversed(self._facts))[:k]


# (statements in order, query, current-value substring, stale-value substring, [noise])
CASES = [
    (["I prefer aisle seats on flights", "I now prefer window seats on flights"],
     "which seat do I prefer on flights?", "window", "aisle"),
    (["My favourite hotel type is budget hostels", "My favourite hotel type is now boutique hotels"],
     "what hotel type do I like?", "boutique", "hostels"),
    (["My phone number is 555-0111", "My phone number changed to 555-0222"],
     "what is my phone number?", "0222", "0111"),
    (["My flight is at 9am", "My flight is now at 2pm"],
     "when is my flight?", "2pm", "9am"),
    (["I work at Acme Corp", "I now work at Globex"],
     "where do I work?", "globex", "acme"),
    (["My meeting room is B12", "My meeting room moved to C04"],
     "which meeting room?", "c04", "b12"),
    (["My diet preference is vegetarian", "My diet preference is now pescatarian"],
     "what is my diet preference?", "pescatarian", "vegetarian"),
    (["My default font size is 12", "My default font size is now 16"],
     "what font size do I use?", "16", "12"),
    (["My car is a red hatchback", "My car is now a blue sedan"],
     "what car do I have?", "sedan", "hatchback"),
    (["My subscription plan is Basic", "My subscription plan upgraded to Pro"],
     "what subscription plan am I on?", "pro", "basic"),
]

NOISE = [
    "I enjoy hiking on weekends.",
    "My cat is named Biscuit.",
    "I studied history at university.",
    "The weather has been rainy lately.",
]


def build(memory_factory, statements):
    m = memory_factory()
    # interleave noise so retrieval is not trivial
    for i, noise in enumerate(NOISE):
        m.add(noise)
    for s in statements:
        m.add(s)
    return m


def score(memory_factory, k=3):
    current_top1 = no_stale = clean = 0
    for statements, query, current, stale in CASES:
        m = build(memory_factory, statements)
        hits = [h.lower() for h in memory_factory_recall(m, query, k)]
        top1 = hits[0] if hits else ""
        has_current = any(current in h for h in hits)
        has_stale = any(stale in h for h in hits)
        if current in top1:
            current_top1 += 1
        if not has_stale:
            no_stale += 1
        if has_current and not has_stale:
            clean += 1
    n = len(CASES)
    return current_top1 / n, no_stale / n, clean / n


def memory_factory_recall(m, query, k):
    out = m.recall(query, k=k)
    return out if isinstance(out, list) else list(out)


def main():
    k = 3
    contenders = {
        "mnem": lambda: Memory(),
        "lexical retriever": LexicalRetriever,
        "recency memory": RecencyMemory,
    }
    print(f"State-correctness benchmark — {len(CASES)} update cases, top-{k} recall\n")
    header = "memory".ljust(20) + "current@1".rjust(12) + "no-stale".rjust(12) + "clean".rjust(10)
    print(header)
    print("-" * len(header))
    for name, factory in contenders.items():
        c1, ns, cl = score(factory, k=k)
        print(name.ljust(20) + f"{100*c1:10.0f}%" + f"{100*ns:11.0f}%" + f"{100*cl:9.0f}%")
    print("-" * len(header))
    print("\nclean = current value present AND stale value suppressed — the outcome that matters.")


if __name__ == "__main__":
    main()
