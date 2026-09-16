"""Local reasoning library: persistent SQLite store of patterns extracted from runs.

A "pattern" is a short, model-readable snippet describing an approach that worked
(or explicitly failed) for a task signature. On a new run, we recall the top-K
patterns whose task signatures look similar to the current task and inject them
as a system note.

This MVP uses simple keyword/LIKE matching. A future version can swap in
embeddings (sqlite-vec) without changing the public interface.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from agentmw.core.embeddings import (
    EmbeddingsBackend,
    NoOpBackend,
    cosine,
    get_default_backend,
    pack_vector,
    unpack_vector,
)


@dataclass
class Pattern:
    id: int
    task_signature: str
    outcome: str  # "success" | "failure"
    pattern_text: str
    created_at: float
    score: float = 0.0  # similarity score when retrieved via semantic recall


def _default_db_path() -> Path:
    base = os.environ.get("AGENTMW_HOME") or os.path.expanduser("~/.agentmw")
    Path(base).mkdir(parents=True, exist_ok=True)
    return Path(base) / "memory.db"


_TOKEN_RE = re.compile(r"[A-Za-z]{3,}")
_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "you", "are", "was",
    "but", "not", "have", "has", "will", "would", "could", "should",
}


def _keywords(text: str, k: int = 6) -> list[str]:
    """Pull a few distinguishing keywords out of a task description."""
    seen: dict[str, int] = {}
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        seen[tok] = seen.get(tok, 0) + 1
    ranked = sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:k]]


def normalize_task(task: str) -> str:
    """Stable, lowercase signature for a task description (used for exact-match fast path)."""
    return hashlib.sha256(" ".join(_keywords(task, 12)).encode("utf-8")).hexdigest()[:16]


class ReasoningLibrary:
    def __init__(
        self,
        db_path: str | Path | None = None,
        embeddings: EmbeddingsBackend | None = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path else _default_db_path()
        self.embeddings: EmbeddingsBackend = embeddings if embeddings is not None else get_default_backend()
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_signature TEXT NOT NULL,
                task_text TEXT NOT NULL,
                outcome TEXT NOT NULL,
                pattern_text TEXT NOT NULL,
                created_at REAL NOT NULL,
                embedding BLOB
            )
            """
        )
        # forward-compatible: older DBs may lack the embedding column
        try:
            self._conn.execute("ALTER TABLE patterns ADD COLUMN embedding BLOB")
        except sqlite3.OperationalError:
            pass
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_patterns_sig ON patterns(task_signature)"
        )
        self._conn.commit()

    @property
    def semantic_enabled(self) -> bool:
        return self.embeddings.available and not isinstance(self.embeddings, NoOpBackend)

    def save(self, task: str, pattern_text: str, outcome: str = "success") -> int:
        sig = normalize_task(task)
        now = time.time()
        emb_blob: bytes | None = None
        if self.semantic_enabled:
            try:
                vec = self.embeddings.embed(task + "\n" + pattern_text, is_query=False)
                emb_blob = pack_vector(vec)
            except Exception:
                emb_blob = None
        cur = self._conn.execute(
            "INSERT INTO patterns(task_signature, task_text, outcome, pattern_text, created_at, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sig, task, outcome, pattern_text, now, emb_blob),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def _recall_semantic(self, task: str, limit: int) -> list[Pattern]:
        try:
            query_vec = self.embeddings.embed(task, is_query=True)
        except Exception:
            return []
        rows = self._conn.execute(
            "SELECT id, task_signature, outcome, pattern_text, created_at, embedding "
            "FROM patterns WHERE embedding IS NOT NULL"
        ).fetchall()
        scored: list[tuple[float, Pattern]] = []
        for r in rows:
            vec = unpack_vector(r[5])
            score = cosine(query_vec, vec)
            scored.append((score, Pattern(id=r[0], task_signature=r[1], outcome=r[2],
                                          pattern_text=r[3], created_at=r[4], score=score)))
        scored.sort(key=lambda kv: -kv[0])
        return [p for s, p in scored[:limit] if s > 0.55]  # threshold cuts garbage matches

    def _recall_keyword(self, task: str, limit: int) -> list[Pattern]:
        sig = normalize_task(task)
        rows = list(
            self._conn.execute(
                "SELECT id, task_signature, outcome, pattern_text, created_at "
                "FROM patterns WHERE task_signature = ? ORDER BY created_at DESC LIMIT ?",
                (sig, limit),
            )
        )
        if len(rows) < limit:
            keywords = _keywords(task, 4)
            if keywords:
                like_clauses = " OR ".join(["task_text LIKE ?"] * len(keywords))
                params = [f"%{kw}%" for kw in keywords] + [limit * 3]
                seen_ids = {r[0] for r in rows}
                extra = self._conn.execute(
                    f"SELECT id, task_signature, outcome, pattern_text, created_at "
                    f"FROM patterns WHERE {like_clauses} ORDER BY created_at DESC LIMIT ?",
                    params,
                ).fetchall()
                for r in extra:
                    if r[0] not in seen_ids and len(rows) < limit:
                        rows.append(r)
                        seen_ids.add(r[0])
        return [Pattern(id=r[0], task_signature=r[1], outcome=r[2], pattern_text=r[3], created_at=r[4]) for r in rows]

    def recall(self, task: str, limit: int = 3) -> list[Pattern]:
        if self.semantic_enabled:
            sem = self._recall_semantic(task, limit)
            if sem:
                return sem
        return self._recall_keyword(task, limit)

    def as_system_note(self, task: str, limit: int = 3) -> str:
        patterns = self.recall(task, limit=limit)
        if not patterns:
            return ""
        lines = ["[agentmw] Relevant reasoning patterns from prior runs:"]
        for p in patterns:
            marker = "✓" if p.outcome == "success" else "✗"
            lines.append(f"  {marker} {p.pattern_text}")
        return "\n".join(lines)

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0])

    def close(self) -> None:
        self._conn.close()
