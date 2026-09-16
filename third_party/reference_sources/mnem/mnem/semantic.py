"""Zero-dependency semantic embedding for mnem.

The usual choices for semantic memory are exact keywords (miss paraphrase) or a
neural embedding model (hundreds of MB, a download, a GPU wish-list). This is a
third option: a **model-free** embedding built from hashed word tokens *and*
character n-grams.

Character n-grams give fuzzy, morphological matching — "colour"/"color",
"run"/"running", "support group"/"LGBTQ support group" — that exact-token overlap
misses, while whole-word features keep precision. It is deterministic, tiny, and
has no dependencies. Not as strong as a trained transformer, but it turns mnem
semantic while still fitting in Redis-sized budgets.

Use it via ``Memory(semantic=True)`` or pass ``embed=HashingEmbedder()`` yourself.
"""

from __future__ import annotations

import math
import re
from hashlib import blake2b

__all__ = ["HashingEmbedder"]

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Hash word tokens and character n-grams into a fixed-dimensional unit vector."""

    def __init__(self, dim: int = 512, min_n: int = 3, max_n: int = 5, word_weight: float = 2.0) -> None:
        self.dim = dim
        self.min_n = min_n
        self.max_n = max_n
        self.word_weight = word_weight

    def __call__(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            self._bump(vec, "w:" + token, self.word_weight)
            padded = "#" + token + "#"
            for n in range(self.min_n, self.max_n + 1):
                for i in range(len(padded) - n + 1):
                    self._bump(vec, padded[i : i + n], 1.0)
        norm = math.sqrt(sum(value * value for value in vec))
        if norm:
            inv = 1.0 / norm
            vec = [value * inv for value in vec]
        return vec

    def _bump(self, vec: list[float], key: str, weight: float) -> None:
        digest = blake2b(key.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % self.dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[index] += sign * weight
