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


class VectorHealth(VectorMemoryBase):
    def _health_unlocked(
        self,
        configured_provider: str,
        configured_dimensions: int,
    ) -> dict[str, Any]:
        _pointer_state, current_id, pointer_reason = self._read_current_pointer_unlocked()
        loaded: _LoadedIndex | None = None
        try:
            loaded = self._load_effective_unlocked(require_compatible=False)
        except Exception:
            loaded = None

        generation_validations = self._valid_published_generations_unlocked(
            require_compatible=False
        )
        generation_count = (
            sum(
                1
                for path in self.generations_path.iterdir()
                if path.is_dir() and self._valid_generation_id(path.name)
            )
            if self.generations_path.is_dir()
            else 0
        )
        previous = next(
            (
                generation_id
                for generation_id, _validation in generation_validations
                if generation_id != (loaded.generation_id if loaded else current_id)
            ),
            None,
        )
        if loaded is None:
            reasons = [pointer_reason] if pointer_reason else []
            if current_id is not None:
                validation = self.validate_generation(
                    self.generations_path / current_id,
                    require_compatible=True,
                )
                reasons.extend(validation.failure_reasons)
            reasons.append("no_valid_generation")
            return self._health_payload(
                status="not_ready",
                storage_mode="corrupt" if generation_count else "empty",
                configured_provider=configured_provider,
                configured_dimensions=configured_dimensions,
                active_generation=current_id,
                previous_generation=previous,
                generation_count=generation_count,
                legacy_detected=self._legacy_detected_unlocked(),
                abandoned_staging_count=len(self._abandoned_staging_ids_unlocked()),
                compatible=not any(
                    reason
                    in {
                        "provider_mismatch",
                        "dimension_mismatch",
                        "embedding_implementation_mismatch",
                    }
                    for reason in reasons
                ),
                failure_reasons=list(dict.fromkeys(reason for reason in reasons if reason)),
            )

        if loaded.storage_mode == "empty":
            return self._health_payload(
                status="ok",
                storage_mode="empty",
                configured_provider=configured_provider,
                configured_dimensions=configured_dimensions,
                generation_count=generation_count,
                legacy_detected=self._legacy_detected_unlocked(),
                abandoned_staging_count=len(self._abandoned_staging_ids_unlocked()),
            )

        metadata = loaded.metadata
        active_provider = metadata.get("provider", self.embedding.name)
        active_dimensions = metadata.get(
            "dimensions",
            int(loaded.vectors.shape[1]) if loaded.vectors.ndim == 2 else None,
        )
        compatible = (
            active_provider == configured_provider
            and active_dimensions == configured_dimensions
            and self._implementation_compatible(metadata)
        )
        fallback = loaded.fallback_active
        legacy = loaded.storage_mode.startswith("legacy")
        degraded = fallback or legacy or not compatible
        reasons = list(loaded.failure_reasons)
        if legacy and "legacy_index_active" not in reasons:
            reasons.append("legacy_index_active")
        if active_provider != configured_provider:
            reasons.append("provider_mismatch")
        if active_dimensions != configured_dimensions:
            reasons.append("dimension_mismatch")
        if not self._implementation_compatible(metadata):
            reasons.append("embedding_implementation_mismatch")
        status = "degraded" if degraded else "ok"
        return self._health_payload(
            status=status,
            storage_mode=loaded.storage_mode,
            configured_provider=configured_provider,
            configured_dimensions=configured_dimensions,
            active_provider=str(active_provider) if active_provider is not None else None,
            active_dimensions=active_dimensions if type(active_dimensions) is int else None,
            active_generation=current_id,
            effective_generation=loaded.generation_id,
            previous_generation=previous,
            generation_count=generation_count,
            record_count=len(loaded.records),
            vector_count=int(loaded.vectors.shape[0]),
            hashes_valid=(
                True
                if legacy
                else bool(
                    self.validate_generation(
                        self.generations_path / str(loaded.generation_id),
                        require_compatible=False,
                    ).hashes_valid
                )
            ),
            compatible=compatible,
            fallback_active=fallback,
            legacy_detected=self._legacy_detected_unlocked(),
            abandoned_staging_count=len(self._abandoned_staging_ids_unlocked()),
            last_successful_reindex_at=metadata.get("last_successful_reindex_at"),
            failure_reasons=reasons,
        )

    def _health_payload(
        self,
        *,
        status: str,
        storage_mode: str,
        configured_provider: str,
        configured_dimensions: int | None,
        active_provider: str | None = None,
        active_dimensions: int | None = None,
        active_generation: str | None = None,
        effective_generation: str | None = None,
        previous_generation: str | None = None,
        generation_count: int = 0,
        record_count: int = 0,
        vector_count: int = 0,
        hashes_valid: bool = False,
        compatible: bool = True,
        fallback_active: bool = False,
        legacy_detected: bool = False,
        abandoned_staging_count: int = 0,
        last_successful_reindex_at: Any = None,
        failure_reasons: list[str] | None = None,
    ) -> dict[str, Any]:
        reasons = list(dict.fromkeys(failure_reasons or []))
        count_match = record_count == vector_count
        return {
            "ok": status == "ok",
            "status": status,
            "storage_mode": storage_mode,
            "configured_provider": configured_provider,
            "configured_implementation_id": self.embedding.implementation_id,
            "active_provider": active_provider,
            "configured_dimensions": configured_dimensions,
            "active_dimensions": active_dimensions,
            "active_generation": active_generation,
            "effective_generation": effective_generation,
            "previous_generation": previous_generation,
            "generation_count": generation_count,
            "record_count": record_count,
            "vector_count": vector_count,
            "record_vector_count_match": count_match,
            "hashes_valid": hashes_valid,
            "compatible": compatible,
            "fallback_active": fallback_active,
            "legacy_index_detected": legacy_detected,
            "abandoned_staging_count": abandoned_staging_count,
            "last_successful_reindex_at": (
                last_successful_reindex_at if isinstance(last_successful_reindex_at, str) else None
            ),
            "failure_reasons": reasons,
            "repair_command": "run april memory repair-index --apply",
            "rebuild_command": "run april memory reindex",
            # Backward-compatible aliases used by existing readiness and tooling.
            "embedding": configured_provider,
            "dimensions": configured_dimensions,
            "persisted_provider": active_provider,
            "persisted_dimensions": active_dimensions,
            "reindex_required": not compatible,
        }

    def _embed_texts(
        self,
        texts: list[str],
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        dimensions = self._configured_dimensions()
        if not texts:
            return _empty_matrix(dimensions)
        matrix = np.empty((len(texts), dimensions), dtype=np.float32)
        completed = 0
        total = len(texts)
        for start in range(0, total, self.embed_batch_size):
            batch = texts[start : start + self.embed_batch_size]
            embedded = np.asarray(self.embedding.embed_many(batch), dtype=np.float32)
            if embedded.ndim != 2 or embedded.shape != (len(batch), dimensions):
                raise ValueError("Embedding provider returned an unexpected batch shape.")
            if not bool(np.isfinite(embedded).all()):
                raise ValueError("Embedding provider returned non-finite values.")
            matrix[start : start + len(batch)] = embedded
            for _text in batch:
                completed += 1
                if progress is not None:
                    progress(completed, total)
        return matrix

    def _configured_dimensions(self) -> int:
        dimensions = self.embedding.dimensions
        if type(dimensions) is not int or dimensions < 1:
            raise ValueError("Embedding dimensions must be a positive integer.")
        return dimensions

    def _legacy_detected_unlocked(self) -> bool:
        return (
            self.records_json_path.exists() and self.vectors_path.exists()
        ) or self.records_path.exists()

    def _raise_incompatible(self, reasons: list[str]) -> None:
        raise ConfigError(
            "Vector index was built with a different embedding configuration. "
            "Refusing to mix vector spaces. Run `run april memory reindex` to rebuild "
            "the index under the current embedding provider.",
            {
                "configured_provider": self.embedding.name,
                "configured_dimensions": self._configured_dimensions(),
                "reason_codes": list(dict.fromkeys(reasons)),
            },
        )

    def _implementation_compatible(self, metadata: dict[str, Any]) -> bool:
        persisted = metadata.get("embedding_implementation_id")
        if isinstance(self.embedding, HashedTokenEmbedding):
            return persisted == self.embedding.implementation_id
        return persisted is None or persisted == self.embedding.implementation_id

    def _audit_recovery(self, generation_id: str, reason: str) -> None:
        if self.audit is None:
            return
        verification = self.audit.verify()
        if not verification.valid:
            issue_codes = ",".join(issue.code for issue in verification.issues)
            issue_codes = issue_codes or verification.status
            raise RuntimeError(
                "Vector index repair requires a writable, valid audit trail; "
                f"audit status={verification.status} ({issue_codes})."
            )
        self.audit.write(
            {
                "event": "memory.vector_index_recovery",
                "generation_id": generation_id,
                "reason_code": reason,
            }
        )
