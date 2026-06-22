from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from april_common.errors import AprilError, ConfigError

if TYPE_CHECKING:
    from april_common.audit import AuditLogger

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensions(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        raise NotImplementedError


class HashedTokenEmbedding(EmbeddingProvider):
    def __init__(self, dimensions: int = 256) -> None:
        self._dimensions = dimensions

    @property
    def name(self) -> str:
        return "hashed-token"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in self._tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimensions
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if math.isclose(norm, 0.0):
            return vector
        return vector / norm

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())


class _RuntimeEmbedClient(Protocol):
    async def embed(self, text: str, *, model_id: str | None = ...) -> list[float]: ...


class RuntimeLocalEmbedding(EmbeddingProvider):
    """Semantic embeddings served by a local embedding-role model via April Runtime.

    The runtime client is injected so this provider never imports model bindings
    directly. Dimensions are resolved once from the first embedding and cached.
    """

    def __init__(self, runtime_client: _RuntimeEmbedClient, model_id: str | None) -> None:
        self._client = runtime_client
        self.model_id = model_id
        self._dimensions: int | None = None

    @property
    def name(self) -> str:
        return "runtime-local"

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            # Resolve lazily from a probe embedding the first time it is needed.
            self.embed("april")
        assert self._dimensions is not None
        return self._dimensions

    def ensure_ready(self) -> int:
        """Probe the runtime once, caching the embedding dimension.

        Raises the underlying AprilError if no embedding model is available so
        callers can decide whether to fall back to a hashed-token provider.
        """
        return self.dimensions

    def embed(self, text: str) -> np.ndarray:
        vector = _run_blocking(self._client.embed(text, model_id=self.model_id))
        array = np.asarray(vector, dtype=np.float32)
        if self._dimensions is None:
            self._dimensions = int(array.shape[0])
        return array


_loop_lock = threading.Lock()
_background_loop: asyncio.AbstractEventLoop | None = None


def _background_event_loop() -> asyncio.AbstractEventLoop:
    """Return a shared, long-lived event loop running on a daemon thread.

    Embeddings are produced from synchronous code (VectorMemory) that may be
    called from inside a running event loop (a FastAPI route) or from plain
    synchronous code. Submitting onto one persistent loop keeps both paths safe
    without repeatedly creating and tearing down event loops.
    """
    global _background_loop
    with _loop_lock:
        if _background_loop is None or _background_loop.is_closed():
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever,
                name="april-embedding-loop",
                daemon=True,
            )
            thread.start()
            _background_loop = loop
        return _background_loop


def _run_blocking(coro: Any) -> Any:
    """Run an awaitable to completion from synchronous code."""
    future = asyncio.run_coroutine_threadsafe(coro, _background_event_loop())
    return future.result()


def embedding_provider_from_config(
    provider: str,
    *,
    model_id: str | None = None,
    runtime_client: _RuntimeEmbedClient | None = None,
    audit: AuditLogger | None = None,
) -> EmbeddingProvider:
    if provider == "hashed-token":
        return HashedTokenEmbedding()
    if provider == "runtime-local":
        return _build_runtime_local(model_id=model_id, runtime_client=runtime_client, audit=audit)
    raise ConfigError(f"Unknown memory embedding provider: {provider}")


def _build_runtime_local(
    *,
    model_id: str | None,
    runtime_client: _RuntimeEmbedClient | None,
    audit: AuditLogger | None,
) -> EmbeddingProvider:
    if runtime_client is None:
        return _fallback(
            reason="no runtime client was provided to resolve embeddings",
            model_id=model_id,
            audit=audit,
        )
    candidate = RuntimeLocalEmbedding(runtime_client, model_id)
    try:
        candidate.ensure_ready()
    except AprilError as exc:
        return _fallback(reason=exc.message, model_id=model_id, audit=audit, error=exc)
    return candidate


def _fallback(
    *,
    reason: str,
    model_id: str | None,
    audit: AuditLogger | None,
    error: AprilError | None = None,
) -> HashedTokenEmbedding:
    message = (
        "memory.embedding_provider=runtime-local requested but the local embedding model is "
        f"unavailable ({reason}); falling back to hashed-token embeddings."
    )
    logger.warning(message)
    if audit is not None:
        audit.write(
            {
                "event": "memory.embedding_fallback",
                "requested_provider": "runtime-local",
                "active_provider": "hashed-token",
                "embedding_model_id": model_id,
                "reason": reason,
                "error_code": error.code if error is not None else None,
            }
        )
    return HashedTokenEmbedding()
