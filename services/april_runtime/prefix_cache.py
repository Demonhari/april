from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class PrefixStateEntry:
    key: tuple[int, ...]
    state: Any
    nbytes: int


class PrefixStatePolicy:
    """Bounded, process-local longest-prefix cache policy."""

    def __init__(self, capacity_bytes: int, min_prefix_tokens: int, max_entries: int) -> None:
        self.capacity_bytes = max(0, capacity_bytes)
        self.min_prefix_tokens = max(0, min_prefix_tokens)
        self.max_entries = max(0, max_entries)
        self._entries: OrderedDict[tuple[int, ...], PrefixStateEntry] = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.lookups = 0
        self.evictions = 0
        self.rejected_oversize = 0

    @property
    def entries(self) -> int:
        return len(self._entries)

    @property
    def bytes(self) -> int:
        return self._bytes

    def stats(self) -> dict[str, int]:
        return {
            "entries": self.entries,
            "bytes": self.bytes,
            "hits": self.hits,
            "misses": self.misses,
            "lookups": self.lookups,
            "evictions": self.evictions,
            "rejected_oversize": self.rejected_oversize,
        }

    def lookup(self, tokens: tuple[int, ...]) -> PrefixStateEntry | None:
        self.lookups += 1
        if not self._entries:
            self.misses += 1
            return None
        query = np.asarray(tokens, dtype=np.int64)
        best: PrefixStateEntry | None = None
        best_length = 0
        for entry in self._entries.values():
            key = np.asarray(entry.key, dtype=np.int64)
            length = min(query.size, key.size)
            if length == 0:
                common = 0
            else:
                mismatch = np.flatnonzero(query[:length] != key[:length])
                common = int(mismatch[0]) if mismatch.size else length
            if common >= self.min_prefix_tokens and common > best_length:
                best = entry
                best_length = common
        if best is None:
            self.misses += 1
            return None
        self._entries.move_to_end(best.key)
        self.hits += 1
        return best

    def insert(self, key: tuple[int, ...], state: Any, nbytes: int) -> bool:
        nbytes = max(0, int(nbytes))
        if nbytes > self.capacity_bytes or self.max_entries == 0:
            self.rejected_oversize += 1
            return False
        old = self._entries.pop(key, None)
        if old is not None:
            self._bytes -= old.nbytes
        self._entries[key] = PrefixStateEntry(key, state, nbytes)
        self._bytes += nbytes
        while self._entries and (
            self._bytes > self.capacity_bytes or len(self._entries) > self.max_entries
        ):
            _key, evicted = self._entries.popitem(last=False)
            self._bytes -= evicted.nbytes
            self.evictions += 1
        return True

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0
