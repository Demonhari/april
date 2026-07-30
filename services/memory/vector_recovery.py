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


class VectorRecovery(VectorMemoryBase):
    def repair_index(self, *, apply: bool = False) -> dict[str, Any]:
        """Inspect or explicitly apply pointer recovery without fabricating vectors."""
        with self._locked():
            pointer_state, current_id, pointer_reason = self._read_current_pointer_unlocked()
            current_validation: GenerationValidationResult | None = None
            if current_id is not None:
                current_validation = self.validate_generation(
                    self.generations_path / current_id,
                    require_compatible=True,
                )
            valid = self._valid_published_generations_unlocked(require_compatible=True)
            candidate = valid[0][0] if valid else None
            retain = self._retained_generation_ids(candidate, valid)
            staging = self._abandoned_staging_ids_unlocked()
            applied = False
            if apply and candidate is not None:
                self._switch_current_unlocked(candidate)
                applied = True
                self._audit_recovery(candidate, pointer_reason or "repair_requested")
                self._cleanup_staging_unlocked()
                self._cleanup_generations_unlocked()
                pointer_state = "valid"
                current_id = candidate
            return {
                "applied": applied,
                "current_pointer_state": pointer_state,
                "current_generation": current_id,
                "current_validation": (
                    {
                        "ok": current_validation.ok,
                        "failure_reasons": list(current_validation.failure_reasons),
                    }
                    if current_validation is not None
                    else None
                ),
                "recovery_candidate": candidate,
                "retained_generations": retain,
                "abandoned_staging_directories": staging,
                "apply_command": "run april memory repair-index --apply",
                "rebuild_command": "run april memory reindex",
                "action": (
                    "repointed"
                    if applied
                    else "dry_run"
                    if candidate is not None
                    else "reindex_required"
                ),
            }

    def _read_index(self) -> _LoadedIndex:
        with self._locked():
            return self._load_effective_unlocked(require_compatible=True)

    def _load_effective_unlocked(self, *, require_compatible: bool) -> _LoadedIndex:
        pointer_state, current_id, pointer_reason = self._read_current_pointer_unlocked()
        failures = [pointer_reason] if pointer_reason else []
        if current_id is not None:
            current_path = self.generations_path / current_id
            validation = self.validate_generation(
                current_path, require_compatible=require_compatible
            )
            if validation.ok:
                return self._load_generation_unlocked(
                    current_id,
                    active_generation=current_id,
                    fallback_active=False,
                    failure_reasons=[],
                )
            failures.extend(validation.failure_reasons)

        for generation_id, _validation in self._valid_published_generations_unlocked(
            require_compatible=require_compatible
        ):
            if generation_id == current_id:
                continue
            loaded = self._load_generation_unlocked(
                generation_id,
                active_generation=current_id,
                fallback_active=True,
                failure_reasons=[
                    *failures,
                    "recovery_generation_active",
                ],
            )
            return loaded

        legacy = self._load_legacy_unlocked(require_compatible=require_compatible)
        if legacy is not None:
            legacy.failure_reasons = [*failures, "legacy_index_active"]
            legacy.active_generation = current_id
            return legacy

        has_index_artifacts = (
            pointer_state != "missing"
            or (self.generations_path.is_dir() and any(self.generations_path.iterdir()))
            or self._legacy_detected_unlocked()
        )
        if has_index_artifacts:
            if require_compatible and (
                "provider_mismatch" in failures
                or "dimension_mismatch" in failures
                or "embedding_implementation_mismatch" in failures
            ):
                self._raise_incompatible(failures)
            raise RuntimeError("No valid vector index generation is available.")
        return _LoadedIndex(
            records=[],
            vectors=_empty_matrix(self._configured_dimensions()),
            storage_mode="empty",
            failure_reasons=failures,
        )

    def _load_generation_unlocked(
        self,
        generation_id: str,
        *,
        active_generation: str | None,
        fallback_active: bool,
        failure_reasons: list[str],
    ) -> _LoadedIndex:
        directory = self.generations_path / generation_id
        records = json.loads((directory / "records.json").read_text(encoding="utf-8"))
        vectors = np.load(directory / "vectors.npy", allow_pickle=False).astype(np.float32)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        return _LoadedIndex(
            records=records,
            vectors=vectors,
            generation_id=generation_id,
            metadata=metadata,
            storage_mode="generation",
            active_generation=active_generation,
            fallback_active=fallback_active,
            failure_reasons=list(dict.fromkeys(reason for reason in failure_reasons if reason)),
        )

    def _load_legacy_unlocked(self, *, require_compatible: bool) -> _LoadedIndex | None:
        if self.records_json_path.is_file() and self.vectors_path.is_file():
            try:
                records = json.loads(self.records_json_path.read_text(encoding="utf-8"))
                vectors = np.load(self.vectors_path, allow_pickle=False)
                metadata = self._legacy_metadata_unlocked()
                self._validate_legacy_data(records, vectors, metadata, require_compatible)
            except ConfigError:
                raise
            except Exception:
                return None
            return _LoadedIndex(
                records=records,
                vectors=vectors.astype(np.float32),
                metadata=metadata,
                storage_mode="legacy-v2",
            )
        if self.records_path.is_file():
            try:
                jsonl_records: list[dict[str, Any]] = []
                rows: list[np.ndarray] = []
                for line in self.records_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    raw = json.loads(line)
                    if not isinstance(raw, dict) or "vector" not in raw:
                        return None
                    record = dict(raw)
                    rows.append(np.asarray(record.pop("vector"), dtype=np.float32))
                    jsonl_records.append(record)
                vectors = _matrix(rows, self._configured_dimensions())
                metadata = self._legacy_metadata_unlocked()
                self._validate_legacy_data(
                    jsonl_records,
                    vectors,
                    metadata,
                    require_compatible,
                )
            except ConfigError:
                raise
            except Exception:
                return None
            return _LoadedIndex(
                records=jsonl_records,
                vectors=vectors,
                metadata=metadata,
                storage_mode="legacy-jsonl",
            )
        return None

    def _validate_legacy_data(
        self,
        records: Any,
        vectors: np.ndarray,
        metadata: dict[str, Any],
        require_compatible: bool,
    ) -> None:
        if not isinstance(records, list) or vectors.ndim != 2 or len(records) != vectors.shape[0]:
            raise RuntimeError("Legacy vector index is invalid.")
        if not bool(np.isfinite(vectors).all()):
            raise RuntimeError("Legacy vector index is invalid.")
        ids: set[str] = set()
        for record in records:
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("id"), str)
                or not record["id"].strip()
                or record["id"] in ids
                or not isinstance(record.get("content"), str)
                or not isinstance(record.get("metadata"), dict)
            ):
                raise RuntimeError("Legacy vector index is invalid.")
            VectorMetadata.model_validate(record["metadata"])
            ids.add(record["id"])
        provider = metadata.get("provider")
        dimensions = metadata.get("dimensions", int(vectors.shape[1]))
        if dimensions != int(vectors.shape[1]):
            raise RuntimeError("Legacy vector index is invalid.")
        if require_compatible and (
            (provider is not None and provider != self.embedding.name)
            or dimensions != self._configured_dimensions()
            or not self._implementation_compatible(metadata)
        ):
            self._raise_incompatible(
                [
                    "provider_mismatch",
                    "dimension_mismatch",
                    "embedding_implementation_mismatch",
                ]
            )

    def _legacy_metadata_unlocked(self) -> dict[str, Any]:
        if not self.metadata_path.is_file() or self.metadata_path.is_symlink():
            return {}
        try:
            value = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
