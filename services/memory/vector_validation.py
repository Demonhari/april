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
from services.memory.vector_base import VectorMemoryBase
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


class VectorValidation(VectorMemoryBase):
    def health(
        self,
        *,
        configured_provider: str | None = None,
        configured_dimensions: int | None = None,
    ) -> dict[str, Any]:
        """Return safe health data even when every on-disk generation is corrupt."""
        provider = configured_provider or self.embedding.name
        if not self.path.exists() and not self._initialize:
            return self._health_payload(
                status="ok",
                storage_mode="empty",
                configured_provider=provider,
                configured_dimensions=configured_dimensions,
            )
        try:
            lock = self._locked() if self._initialize else self._inspection_locked()
            with lock:
                dimensions = (
                    configured_dimensions
                    if configured_dimensions is not None
                    else self._configured_dimensions()
                )
                return self._health_unlocked(provider, dimensions)
        except Exception:
            return self._health_payload(
                status="not_ready",
                storage_mode="unavailable",
                configured_provider=provider,
                configured_dimensions=configured_dimensions,
                failure_reasons=["no_valid_generation"],
            )

    def validate_generation(
        self,
        generation_path: Path,
        *,
        require_compatible: bool = True,
        published: bool = True,
    ) -> GenerationValidationResult:
        """Validate one complete generation without exposing record contents."""
        root = self.path.resolve(strict=False)
        candidate = generation_path.resolve(strict=False)
        generation_id = generation_path.name
        reasons: list[str] = []
        metadata: dict[str, Any] = {}
        records: list[dict[str, Any]] = []
        vectors: np.ndarray | None = None
        hashes_valid = False
        provider: str | None = None
        dimensions: int | None = None

        if not _is_beneath(candidate, root) or not self._valid_generation_id(generation_id):
            return GenerationValidationResult(False, generation_id, ("generation_path_invalid",))
        if generation_path.is_symlink() or not generation_path.is_dir():
            return GenerationValidationResult(False, generation_id, ("current_generation_missing",))
        paths = {name: generation_path / name for name in _REQUIRED_FILES}
        if any(path.is_symlink() or not path.is_file() for path in paths.values()):
            return GenerationValidationResult(False, generation_id, ("generation_files_missing",))

        try:
            raw_metadata = json.loads(paths["metadata.json"].read_text(encoding="utf-8"))
            if not isinstance(raw_metadata, dict):
                reasons.append("generation_metadata_invalid")
            else:
                metadata = raw_metadata
        except (OSError, UnicodeError, json.JSONDecodeError):
            reasons.append("generation_metadata_invalid")

        try:
            raw_records = json.loads(paths["records.json"].read_text(encoding="utf-8"))
            if not isinstance(raw_records, list):
                reasons.append("records_invalid")
            else:
                records = raw_records
        except (OSError, UnicodeError, json.JSONDecodeError):
            reasons.append("records_invalid")

        record_ids: set[str] = set()
        if records:
            for record in records:
                if not isinstance(record, dict):
                    reasons.append("records_invalid")
                    break
                record_id = record.get("id")
                content = record.get("content")
                record_metadata = record.get("metadata")
                if (
                    not isinstance(record_id, str)
                    or not record_id.strip()
                    or record_id in record_ids
                    or not isinstance(content, str)
                    or not isinstance(record_metadata, dict)
                ):
                    reasons.append("records_invalid")
                    break
                try:
                    VectorMetadata.model_validate(record_metadata)
                except Exception:
                    reasons.append("records_invalid")
                    break
                record_ids.add(record_id)

        try:
            loaded = np.load(paths["vectors.npy"], allow_pickle=False)
            if (
                not isinstance(loaded, np.ndarray)
                or loaded.ndim != 2
                or not np.issubdtype(loaded.dtype, np.number)
            ):
                reasons.append("vectors_invalid")
            else:
                vectors = loaded
                if not bool(np.isfinite(vectors).all()):
                    reasons.append("vectors_non_finite")
        except (OSError, ValueError, EOFError):
            reasons.append("vectors_invalid")

        if metadata:
            provider_value = metadata.get("provider")
            dimensions_value = metadata.get("dimensions")
            provider = provider_value if isinstance(provider_value, str) else None
            dimensions = dimensions_value if type(dimensions_value) is int else None
            if metadata.get("format_version") != FORMAT_VERSION:
                reasons.append("format_version_unsupported")
            if metadata.get("generation_id") != generation_id:
                reasons.append("generation_id_mismatch")
            if not provider or not provider.strip() or dimensions is None or dimensions < 1:
                reasons.append("generation_metadata_invalid")
            if (
                not isinstance(metadata.get("created_at"), str)
                or not str(metadata["created_at"]).strip()
            ):
                reasons.append("generation_metadata_invalid")
            if (
                type(metadata.get("record_count")) is not int
                or int(metadata["record_count"]) < 0
                or type(metadata.get("vector_count")) is not int
                or int(metadata["vector_count"]) < 0
            ):
                reasons.append("generation_metadata_invalid")
            source_generation = metadata.get("source_generation_id")
            if source_generation is not None and (
                not isinstance(source_generation, str)
                or not self._valid_generation_id(source_generation)
            ):
                reasons.append("generation_metadata_invalid")
            last_reindex = metadata.get("last_successful_reindex_at")
            if last_reindex is not None and not isinstance(last_reindex, str):
                reasons.append("generation_metadata_invalid")
            if published and metadata.get("failure_state") is not None:
                reasons.append("generation_failure_state")

        vector_count = int(vectors.shape[0]) if vectors is not None and vectors.ndim == 2 else 0
        vector_dimensions = (
            int(vectors.shape[1]) if vectors is not None and vectors.ndim == 2 else None
        )
        if vectors is not None and vector_count != len(records):
            reasons.append("record_vector_count_mismatch")
        if (
            dimensions is not None
            and vector_dimensions is not None
            and dimensions != vector_dimensions
        ):
            reasons.append("dimension_mismatch")
        if metadata:
            if metadata.get("record_count") != len(records):
                reasons.append("record_vector_count_mismatch")
            if metadata.get("vector_count") != vector_count:
                reasons.append("record_vector_count_mismatch")
            try:
                records_hash = _sha256_file(paths["records.json"])
                vectors_hash = _sha256_file(paths["vectors.npy"])
                hashes_valid = (
                    metadata.get("records_sha256") == records_hash
                    and metadata.get("vectors_sha256") == vectors_hash
                )
            except OSError:
                hashes_valid = False
            if not hashes_valid:
                reasons.append("generation_hash_mismatch")

        compatible = (
            provider == self.embedding.name
            and dimensions is not None
            and dimensions == self._configured_dimensions()
            and self._implementation_compatible(metadata)
        )
        if require_compatible and provider is not None and provider != self.embedding.name:
            reasons.append("provider_mismatch")
        if (
            require_compatible
            and dimensions is not None
            and dimensions != self._configured_dimensions()
        ):
            reasons.append("dimension_mismatch")
        if require_compatible and metadata and not self._implementation_compatible(metadata):
            reasons.append("embedding_implementation_mismatch")

        unique_reasons = tuple(dict.fromkeys(reasons))
        return GenerationValidationResult(
            ok=not unique_reasons,
            generation_id=generation_id,
            failure_reasons=unique_reasons,
            record_count=len(records),
            vector_count=vector_count,
            dimensions=dimensions,
            provider=provider,
            hashes_valid=hashes_valid,
            compatible=compatible,
            metadata=self._safe_metadata(metadata),
        )
