"""Backward-compatible facade for the generation-based vector index."""

from __future__ import annotations

# Public constants and result models are compatibility re-exports.
from services.memory.vector_health import VectorHealth
from services.memory.vector_indexing import VectorIndexing
from services.memory.vector_models import (
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_RETENTION,
    FORMAT_VERSION,
    MAX_EMBED_BATCH_SIZE,
    GenerationValidationResult,
)
from services.memory.vector_publication import VectorPublication
from services.memory.vector_recovery import VectorRecovery
from services.memory.vector_validation import VectorValidation


class VectorMemory(
    VectorValidation,
    VectorIndexing,
    VectorRecovery,
    VectorPublication,
    VectorHealth,
):
    """Stable public facade retaining the staged-generation implementation."""


__all__ = [
    "DEFAULT_EMBED_BATCH_SIZE",
    "DEFAULT_RETENTION",
    "FORMAT_VERSION",
    "MAX_EMBED_BATCH_SIZE",
    "GenerationValidationResult",
    "VectorMemory",
]
