"""Pluggable embedding backends for the reasoning library.

Default backend is a no-op so the core package stays dependency-free.
Install `agentmw[semantic]` to enable `FastEmbedBackend` (30-50 MB ONNX model,
runs on CPU, no API key).
"""

from __future__ import annotations

import struct
from typing import Protocol


class EmbeddingsBackend(Protocol):
    available: bool
    dim: int

    def embed(self, text: str, *, is_query: bool = False) -> list[float]: ...


class NoOpBackend:
    available = False
    dim = 0

    def embed(self, text: str, *, is_query: bool = False) -> list[float]:
        raise RuntimeError("NoOpBackend has no embeddings; install agentmw[semantic].")


# BGE-style models want different prefixes for retrieval queries vs. stored passages.
# Without the prefix the cosine of two unrelated texts sits around 0.5,
# which produces too many false positives.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class FastEmbedBackend:
    """ONNX-runtime backed BGE-small (384 dim). Lazy-init on first embed."""

    available = True

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model = None
        self.dim = 384  # bge-small-en-v1.5
        self._uses_bge_prefix = "bge" in model_name.lower()

    def _ensure(self) -> None:
        if self._model is None:
            from fastembed import TextEmbedding  # lazy import
            self._model = TextEmbedding(self.model_name)

    def embed(self, text: str, *, is_query: bool = False) -> list[float]:
        self._ensure()
        assert self._model is not None
        prefixed = (_BGE_QUERY_PREFIX + text) if (is_query and self._uses_bge_prefix) else text
        vecs = list(self._model.embed([prefixed]))
        v = vecs[0]
        return [float(x) for x in v]


def get_default_backend() -> EmbeddingsBackend:
    """Best available backend; never raises."""
    try:
        import fastembed  # noqa: F401
        return FastEmbedBackend()
    except Exception:
        return NoOpBackend()


def pack_vector(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_vector(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))
