from __future__ import annotations

from april_common.audit import AuditLogger
from april_common.settings import AprilSettings
from services.memory.embeddings import (
    EmbeddingProvider,
    embedding_provider_from_config,
)
from services.memory.embeddings import (
    _RuntimeEmbedClient as RuntimeEmbedClient,
)
from services.memory.vector_memory import VectorMemory


def configured_embedding_provider(
    settings: AprilSettings,
    *,
    runtime_client: RuntimeEmbedClient | None = None,
    audit: AuditLogger | None = None,
) -> EmbeddingProvider:
    """Resolve the embedding provider exactly as the Core API container does.

    Tools (``repo_indexer``, ``document_indexer``, ``document_search``) execute
    inside the Core API process but historically built their own
    ``VectorMemory(settings.vector_index_path)`` handle, which silently defaulted
    to hashed-token embeddings regardless of ``memory.embedding_provider``. That
    meant a tool could write/read a *different* vector space than the container,
    quietly mixing incompatible embeddings.

    Routing every tool-side construction through this helper keeps all readers and
    writers of the local vector index on the SAME configured embedding space. For
    ``runtime-local`` a runtime client is required to reach the local embedding
    model; if one is not supplied we build the standard April Runtime HTTP client
    from settings (models still only ever go through ``services/april_runtime``).
    When the local embedding model cannot be reached, ``embedding_provider_from_config``
    applies the existing audited fallback policy to hashed-token rather than
    failing the tool — never a silent space switch.
    """
    resolved_client = runtime_client
    if resolved_client is None and settings.memory.embedding_provider == "runtime-local":
        # Lazy import keeps the memory package free of an import-time dependency on
        # the runtime client; the client is a thin authenticated HTTP wrapper.
        from services.april_runtime.client import RuntimeClient

        resolved_client = RuntimeClient(
            settings.runtime.url,
            timeout=settings.runtime.request_timeout_seconds,
            token=settings.runtime.token,
        )
    return embedding_provider_from_config(
        settings.memory.embedding_provider,
        model_id=settings.memory.embedding_model_id,
        runtime_client=resolved_client,
        audit=audit,
    )


def vector_memory_from_settings(
    settings: AprilSettings,
    *,
    runtime_client: RuntimeEmbedClient | None = None,
    audit: AuditLogger | None = None,
) -> VectorMemory:
    """Build a ``VectorMemory`` bound to the configured embedding provider.

    This is the single seam tools use so they never silently fall back to
    hashed-token while the container serves runtime-local vectors (or vice versa).
    The persisted index header still enforces compatibility; this helper makes the
    *active* provider match the container's configuration in the first place.
    """
    embedding = configured_embedding_provider(
        settings,
        runtime_client=runtime_client,
        audit=audit,
    )
    return VectorMemory(settings.vector_index_path, embedding=embedding)
