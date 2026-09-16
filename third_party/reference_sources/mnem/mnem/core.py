"""mnem — memory as state, not search.

Most memory systems answer *"what is similar?"*. An agent usually needs
*"what is true now?"*. mnem models memory as evolving **state**: every statement
is bucketed into a topic, a newer statement about a topic **supersedes** the
older value, stale values are suppressed, and private facts never surface.

It is deterministic and dependency-free. Token salience is learned from the
corpus itself (an IDF weight) instead of a hand-written stop-word list, so it
adapts to any domain or language. Plug an optional embedding function to swap
the lexical topic model for a semantic one.
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

__all__ = ["Memory", "Fact"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class Fact:
    """A single remembered statement and its lifecycle state."""

    text: str
    timestamp: float
    source: Optional[str] = None
    private: bool = False
    reinforce: int = 1
    removed: bool = False
    superseded_by: Optional[int] = None
    tokens: set[str] = field(default_factory=set)

    @property
    def current(self) -> bool:
        return not self.private and not self.removed and self.superseded_by is None


class Memory:
    """An in-process, deterministic state memory.

    Example
    -------
    >>> m = Memory()
    >>> m.add("I prefer aisle seats on flights")
    0
    >>> m.add("Actually I now prefer window seats")   # supersedes the aisle fact
    1
    >>> m.recall("which seat do I like?")
    ['Actually I now prefer window seats']

    Parameters
    ----------
    supersede_threshold:
        How much two statements must overlap in topic (0..1) for the newer one
        to replace the older. Lower = more aggressive supersession.
    weights:
        Blend of (relevance, recency, reinforcement) used to rank recall.
    embed:
        Optional ``str -> sequence[float]`` function. When given, topic matching
        and relevance use cosine similarity of embeddings instead of lexical
        IDF overlap — turning mnem into a semantic memory while keeping the same
        state/supersession logic. Leave ``None`` for the zero-dependency mode.
    """

    def __init__(
        self,
        *,
        supersede_threshold: float = 0.55,
        weights: tuple[float, float, float] = (0.6, 0.25, 0.15),
        embed: Optional[Callable[[str], Sequence[float]]] = None,
        semantic: bool = False,
    ) -> None:
        if not 0.0 < supersede_threshold <= 1.0:
            raise ValueError("supersede_threshold must be in (0, 1]")
        self.supersede_threshold = supersede_threshold
        self.weights = weights
        if embed is None and semantic:
            from .semantic import HashingEmbedder  # zero-dependency, built in

            embed = HashingEmbedder()
        self._embed = embed
        self._facts: list[Fact] = []
        self._vectors: dict[int, Sequence[float]] = {}
        self._doc_freq: dict[str, int] = {}
        self._doc_count = 0
        # inverted index: token -> ids of *current* facts containing it. Keeps
        # supersession near-linear by only comparing facts that share a token.
        self._postings: dict[str, set[int]] = {}
        # tokens shared by more than this many current facts are non-distinctive
        # (stop-word-like) and skipped when gathering supersession candidates.
        self._candidate_cap = 200
        # Markdown persistence (see mnem/store.py); bound via Memory.open().
        self._store_path: Optional[str] = None
        self._autosave = False

    # ------------------------------------------------------------------ writes

    def add(
        self,
        text: str,
        *,
        timestamp: Optional[float] = None,
        source: Optional[str] = None,
        private: bool = False,
        reinforce: bool = True,
    ) -> Optional[int]:
        """Remember ``text``. Returns the fact's id, or ``None`` if empty.

        A non-private statement whose topic matches an existing current fact
        supersedes it (the old value stops being returned). Private facts are
        stored but never surfaced by :meth:`recall` or :meth:`state`.
        """
        text = " ".join(str(text).split())
        if not text:
            return None
        ts = time.time() if timestamp is None else float(timestamp)
        tokens = set(_tokenize(text))
        fact = Fact(text=text, timestamp=ts, source=source, private=private, tokens=tokens)
        idx = len(self._facts)
        self._facts.append(fact)
        if self._embed is not None:
            self._vectors[idx] = list(self._embed(text))
        if private:
            self._maybe_autosave()
            return idx

        self._doc_count += 1
        for token in tokens:
            self._doc_freq[token] = self._doc_freq.get(token, 0) + 1

        if reinforce:
            if self._embed is None:
                candidates: set[int] = set()
                for token in tokens:
                    bucket = self._postings.get(token)
                    if bucket and len(bucket) <= self._candidate_cap:
                        candidates |= bucket
            else:  # semantic mode has no token index; fall back to current facts
                candidates = {j for j in range(idx) if self._facts[j].current}
            best, best_sim = None, 0.0
            for other_idx in candidates:
                sim = self._topic_sim(idx, other_idx)
                if sim > best_sim:
                    best, best_sim = other_idx, sim
            if best is not None and best_sim >= self.supersede_threshold:
                self._facts[best].superseded_by = idx
                fact.reinforce = self._facts[best].reinforce + 1
                self._unindex(best)

        for token in tokens:
            self._postings.setdefault(token, set()).add(idx)
        self._maybe_autosave()
        return idx

    def extend(self, texts: Iterable[str], **kwargs) -> list[Optional[int]]:
        """Add many statements in order."""
        return [self.add(text, **kwargs) for text in texts]

    def forget(self, needle: str) -> int:
        """Remove every stored fact containing ``needle`` (case-insensitive)."""
        needle = needle.lower()
        removed = 0
        for idx, fact in enumerate(self._facts):
            if not fact.removed and needle in fact.text.lower():
                fact.removed = True
                self._unindex(idx)
                removed += 1
        if removed:
            self._maybe_autosave()
        return removed

    def _unindex(self, idx: int) -> None:
        for token in self._facts[idx].tokens:
            bucket = self._postings.get(token)
            if bucket is not None:
                bucket.discard(idx)

    # ------------------------------------------------------------------- reads

    def state(self) -> list[str]:
        """The current facts (newest topic value first), stale ones dropped."""
        current = [fact for fact in self._facts if fact.current]
        current.sort(key=lambda fact: fact.timestamp, reverse=True)
        return [fact.text for fact in current]

    def recall(self, query: str, k: int = 5) -> list[str]:
        """Return up to ``k`` current facts most relevant to ``query``."""
        return [fact.text for fact in self.recall_facts(query, k)]

    def recall_facts(self, query: str, k: int = 5) -> list[Fact]:
        current = [(idx, fact) for idx, fact in enumerate(self._facts) if fact.current]
        if not current:
            return []
        query_tokens = set(_tokenize(query))
        timestamps = [fact.timestamp for _, fact in current]
        earliest, latest = min(timestamps), max(timestamps)
        span = (latest - earliest) or 1.0
        w_rel, w_rec, w_rein = self.weights
        scored = []
        for idx, fact in current:
            relevance = self._relevance(idx, query, query_tokens)
            recency = (fact.timestamp - earliest) / span
            reinforcement = min(fact.reinforce, 5) / 5.0
            score = w_rel * relevance + w_rec * recency + w_rein * reinforcement
            scored.append((score, recency, fact))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [fact for _, _, fact in scored[:k]]

    def facts(self, *, include_private: bool = False) -> list[Fact]:
        """All stored facts (history included). Private ones hidden by default."""
        return [f for f in self._facts if include_private or not f.private]

    def __len__(self) -> int:
        return sum(1 for fact in self._facts if fact.current)

    def prompt(
        self,
        query: Optional[str] = None,
        *,
        budget: int = 800,
        header: str = "Current facts (stale values already superseded):",
    ) -> str:
        """Compile the current state into an LLM-ready block of ~``budget`` tokens.

        Paste the result into a system prompt. With ``query`` the facts are
        ranked by relevance; without it, newest first. Token cost is estimated
        at ~4 characters per token.
        """
        if query:
            facts = self.recall_facts(query, k=max(len(self._facts), 1))
        else:
            facts = sorted((f for f in self._facts if f.current), key=lambda f: f.timestamp, reverse=True)
        used = len(header) // 4
        lines: list[str] = []
        for fact in facts:
            cost = len(fact.text) // 4 + 2
            if lines and used + cost > budget:
                break
            lines.append(f"- {fact.text}")
            used += cost
        if not lines:
            return ""
        return header + "\n" + "\n".join(lines)

    # ----------------------------------------------------------------- file

    @classmethod
    def open(cls, path, *, autosave: bool = True, **kwargs) -> "Memory":
        """Open (or create) a Markdown-backed memory: the file IS the database.

        Loads existing facts, then keeps the file in sync after every ``add``
        and ``forget`` (unless ``autosave=False``). Edit the file by hand and
        re-open it: your edits are the new truth.
        """
        from . import store

        memory = cls(**kwargs)
        memory._store_path = str(path)
        if os.path.exists(path):
            store.load_into(memory, path)
        memory._autosave = autosave
        if autosave:
            memory.save()
        return memory

    def save(self, path=None, *, include_private: bool = False) -> str:
        """Write the memory to its Markdown file (or ``path``). Returns the path."""
        from . import store

        target = path or self._store_path
        if target is None:
            raise ValueError("This memory is not file-bound; pass save(path=...).")
        store.dump_to(self, target, include_private=include_private)
        if self._store_path is None:
            self._store_path = str(target)
        return str(target)

    def _maybe_autosave(self) -> None:
        if self._autosave and self._store_path:
            self.save()

    # -------------------------------------------------------------- internals

    def _idf(self, token: str) -> float:
        return math.log((1 + self._doc_count) / (1 + self._doc_freq.get(token, 0))) + 1.0

    def _topic_sim(self, i: int, j: int) -> float:
        if self._embed is not None:
            return _cosine(self._vectors[i], self._vectors[j])
        left, right = self._facts[i].tokens, self._facts[j].tokens
        shared = left & right
        if not shared:
            return 0.0
        num = sum(self._idf(token) for token in shared)
        den = min(
            sum(self._idf(token) for token in left),
            sum(self._idf(token) for token in right),
        ) or 1.0
        return num / den

    def _relevance(self, idx: int, query: str, query_tokens: set[str]) -> float:
        if self._embed is not None:
            return max(0.0, _cosine(self._vectors[idx], list(self._embed(query))))
        shared = self._facts[idx].tokens & query_tokens
        if not shared:
            return 0.0
        num = sum(self._idf(token) for token in shared)
        den = sum(self._idf(token) for token in query_tokens) or 1.0
        return num / den
