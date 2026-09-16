"""Test that semantic recall finds patterns even when wording is different."""

import os
import tempfile

import pytest

fastembed = pytest.importorskip("fastembed")  # skip suite if not installed

from agentmw.core.embeddings import FastEmbedBackend
from agentmw.core.memory import ReasoningLibrary


def test_semantic_recall_across_paraphrase():
    """Pattern saved with one phrasing must be recalled when the task is paraphrased."""
    with tempfile.TemporaryDirectory() as tmp:
        lib = ReasoningLibrary(db_path=os.path.join(tmp, "m.db"), embeddings=FastEmbedBackend())
        assert lib.semantic_enabled

        lib.save(
            task="Find why the user-creation flow drops the email field on retry",
            pattern_text="Check for `.pop('email')` or `del payload['email']` in retry handlers.",
            outcome="success",
        )
        lib.save(
            task="Investigate slow queries on the orders table",
            pattern_text="Add an index on (customer_id, created_at) — covered 4/5 cases.",
            outcome="success",
        )

        # Same intent, completely different wording
        results = lib.recall("email is missing when account creation is retried", limit=2)
        assert len(results) >= 1
        top = results[0]
        assert "pop" in top.pattern_text.lower() or "email" in top.pattern_text.lower()
        assert top.score > 0.55, f"top score too low: {top.score}"

        # A totally unrelated query should fall below the recall threshold entirely
        unrelated = lib.recall("how do I deploy a React app to Vercel", limit=3)
        assert unrelated == [], f"unrelated query returned {len(unrelated)} false matches"

        lib.close()


def test_keyword_fallback_when_no_embeddings(monkeypatch):
    """If embedding backend is the no-op, keyword recall must still work."""
    from agentmw.core.embeddings import NoOpBackend

    with tempfile.TemporaryDirectory() as tmp:
        lib = ReasoningLibrary(db_path=os.path.join(tmp, "m.db"), embeddings=NoOpBackend())
        assert not lib.semantic_enabled
        lib.save("fix the broken login button", "check event handler bindings")
        results = lib.recall("broken login button fix")
        assert len(results) >= 1
        lib.close()
