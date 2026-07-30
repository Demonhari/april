from __future__ import annotations

# ruff: noqa: F401
# mypy: disable-error-code="attr-defined"
import contextlib
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from april_common.errors import ConfigError
from april_common.time import utc_now_iso
from services.memory.embeddings import EmbeddingProvider, HashedTokenEmbedding
from services.memory.schemas import SearchResult, VectorMetadata
from services.memory.vector_models import (
    _DIRECTORY_FSYNC_UNSUPPORTED,
    _GENERATION_RE,
    _REQUIRED_FILES,
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_RETENTION,
    FORMAT_VERSION,
    MAX_EMBED_BATCH_SIZE,
    GenerationValidationResult,
    _empty_matrix,
    _fsync_directory,
    _is_beneath,
    _LoadedIndex,
    _matrix,
    _process_lock,
    _sha256_file,
)

if TYPE_CHECKING:
    from april_common.audit import AuditLogger


class VectorMemoryBase:
    def __init__(
        self,
        path: Path,
        embedding: EmbeddingProvider | None = None,
        *,
        audit: AuditLogger | None = None,
        retention: int = DEFAULT_RETENTION,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        initialize: bool = True,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.embedding = embedding or HashedTokenEmbedding()
        self.audit = audit
        self.retention = max(2, retention)
        self.embed_batch_size = max(1, min(embed_batch_size, MAX_EMBED_BATCH_SIZE))
        self._initialize = initialize
        self.current_path = self.path / "CURRENT"
        self.generations_path = self.path / "generations"
        self.staging_path = self.path / "staging"
        self.lock_path = self.path / ".lock"
        # Legacy root-file locations stay readable until a successful mutation.
        self.vectors_path = self.path / "vectors.npy"
        self.records_json_path = self.path / "records.json"
        self.records_path = self.path / "records.jsonl"
        self.metadata_path = self.path / "metadata.json"
        if initialize:
            self.path.mkdir(parents=True, exist_ok=True)
            self.generations_path.mkdir(exist_ok=True)
            self.staging_path.mkdir(exist_ok=True)

    @staticmethod
    def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "format_version",
            "generation_id",
            "provider",
            "embedding_implementation_id",
            "embedding_model_id",
            "dimensions",
            "record_count",
            "vector_count",
            "records_sha256",
            "vectors_sha256",
            "created_at",
            "last_successful_reindex_at",
            "source_generation_id",
            "failure_state",
        }
        return {key: value for key, value in metadata.items() if key in allowed}

    @staticmethod
    def _valid_generation_id(generation_id: str) -> bool:
        return bool(_GENERATION_RE.fullmatch(generation_id))

    @staticmethod
    def _new_generation_id() -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        return f"g-{timestamp}-{uuid.uuid4().hex[:12]}"

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.mkdir(parents=True, exist_ok=True)
        self.generations_path.mkdir(exist_ok=True)
        self.staging_path.mkdir(exist_ok=True)
        process_lock = _process_lock(self.path)
        with process_lock, self.lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @contextlib.contextmanager
    def _inspection_locked(self) -> Iterator[None]:
        """Use an existing advisory lock without creating files during doctor."""
        process_lock = _process_lock(self.path)
        with process_lock:
            if self.lock_path.is_file() and not self.lock_path.is_symlink():
                with self.lock_path.open("rb") as handle:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    try:
                        yield
                    finally:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            else:
                yield
