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


class VectorPublication(VectorMemoryBase):
    def _publish_unlocked(
        self,
        records: list[dict[str, Any]],
        vectors: np.ndarray,
        *,
        source: _LoadedIndex,
        last_successful_reindex_at: str | None = None,
    ) -> str:
        dimensions = self._configured_dimensions()
        normalized = np.asarray(vectors, dtype=np.float32)
        if normalized.ndim != 2 or normalized.shape != (len(records), dimensions):
            raise ValueError("Vector matrix shape does not match the generation metadata.")
        if not bool(np.isfinite(normalized).all()):
            raise ValueError("Vector matrix contains non-finite values.")

        generation_id = self._new_generation_id()
        staging = self.staging_path / generation_id
        published = self.generations_path / generation_id
        staging.mkdir()
        try:
            records_path = staging / "records.json"
            vectors_path = staging / "vectors.npy"
            metadata_path = staging / "metadata.json"
            self._write_text_file(
                records_path, json.dumps(records, sort_keys=True, separators=(",", ":")) + "\n"
            )
            self._write_vector_file(vectors_path, normalized)
            metadata: dict[str, Any] = {
                "format_version": FORMAT_VERSION,
                "generation_id": generation_id,
                "provider": self.embedding.name,
                "embedding_implementation_id": self.embedding.implementation_id,
                "dimensions": dimensions,
                "record_count": len(records),
                "vector_count": int(normalized.shape[0]),
                "records_sha256": _sha256_file(records_path),
                "vectors_sha256": _sha256_file(vectors_path),
                "created_at": utc_now_iso(),
                "last_successful_reindex_at": (
                    last_successful_reindex_at or source.metadata.get("last_successful_reindex_at")
                ),
                "source_generation_id": source.generation_id,
                "failure_state": None,
            }
            model_id = getattr(self.embedding, "model_id", None)
            if isinstance(model_id, str) and model_id:
                metadata["embedding_model_id"] = model_id
            self._write_text_file(
                metadata_path,
                json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
            )
            _fsync_directory(staging)
            staging_validation = self.validate_generation(
                staging, require_compatible=True, published=False
            )
            if not staging_validation.ok:
                raise RuntimeError(
                    "Staged vector generation validation failed: "
                    + ",".join(staging_validation.failure_reasons)
                )
            self._publish_staging_directory(staging, published)
            _fsync_directory(self.generations_path)
            published_validation = self.validate_generation(
                published, require_compatible=True, published=True
            )
            if not published_validation.ok:
                raise RuntimeError(
                    "Published vector generation validation failed: "
                    + ",".join(published_validation.failure_reasons)
                )
            self._switch_current_unlocked(generation_id)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

        # Publication and pointer switch are already durable. Retention is
        # deliberately best-effort so cleanup cannot turn success into failure.
        with contextlib.suppress(OSError, RuntimeError):
            self._cleanup_generations_unlocked()
        return generation_id

    def _write_text_file(self, path: Path, content: str) -> None:
        with path.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

    def _write_vector_file(self, path: Path, vectors: np.ndarray) -> None:
        with path.open("wb") as handle:
            np.save(handle, vectors.astype(np.float32), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())

    def _publish_staging_directory(self, staging: Path, published: Path) -> None:
        if published.exists() or published.is_symlink():
            raise FileExistsError("Vector generation identifier already exists.")
        os.rename(staging, published)

    def _switch_current_unlocked(self, generation_id: str) -> None:
        if not self._valid_generation_id(generation_id):
            raise ValueError("Invalid vector generation identifier.")
        descriptor, raw_temp = tempfile.mkstemp(dir=self.path, prefix=".CURRENT.", text=True)
        temp = Path(raw_temp)
        try:
            self._write_current_temp_file(descriptor, generation_id)
            self._replace_current_pointer(temp)
            _fsync_directory(self.path)
        finally:
            if temp.exists():
                temp.unlink()

    def _write_current_temp_file(self, descriptor: int, generation_id: str) -> None:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"{generation_id}\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _replace_current_pointer(self, temp: Path) -> None:
        os.replace(temp, self.current_path)

    def _read_current_pointer_unlocked(self) -> tuple[str, str | None, str | None]:
        if not self.current_path.exists() and not self.current_path.is_symlink():
            return "missing", None, "current_missing"
        if self.current_path.is_symlink() or not self.current_path.is_file():
            return "malformed", None, "current_malformed"
        try:
            raw = self.current_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return "malformed", None, "current_malformed"
        if not raw.endswith("\n") or raw.count("\n") != 1:
            return "malformed", None, "current_malformed"
        generation_id = raw[:-1]
        if not self._valid_generation_id(generation_id):
            return "malformed", None, "current_malformed"
        if not (self.generations_path / generation_id).is_dir():
            return "missing_generation", generation_id, "current_generation_missing"
        return "valid", generation_id, None

    def _valid_published_generations_unlocked(
        self, *, require_compatible: bool
    ) -> list[tuple[str, GenerationValidationResult]]:
        results: list[tuple[str, GenerationValidationResult]] = []
        if not self.generations_path.is_dir():
            return results
        for path in sorted(
            self.generations_path.iterdir(),
            key=lambda item: item.name,
            reverse=True,
        ):
            if not self._valid_generation_id(path.name):
                continue
            validation = self.validate_generation(
                path, require_compatible=require_compatible, published=True
            )
            if validation.ok:
                results.append((path.name, validation))
        return results

    def _cleanup_generations_unlocked(self) -> None:
        state, current_id, _reason = self._read_current_pointer_unlocked()
        if state != "valid" or current_id is None:
            return
        current_validation = self.validate_generation(
            self.generations_path / current_id, require_compatible=True
        )
        if not current_validation.ok:
            return
        valid = self._valid_published_generations_unlocked(require_compatible=False)
        keep = set(self._retained_generation_ids(current_id, valid))
        for generation_path in self.generations_path.iterdir():
            generation_id = generation_path.name
            if (
                self._valid_generation_id(generation_id)
                and generation_id not in keep
                and generation_id != current_id
                and generation_path.is_dir()
                and not generation_path.is_symlink()
            ):
                shutil.rmtree(generation_path)
        _fsync_directory(self.generations_path)

    def _retained_generation_ids(
        self,
        current_id: str | None,
        valid: list[tuple[str, GenerationValidationResult]],
    ) -> list[str]:
        retained: list[str] = []
        if current_id is not None:
            retained.append(current_id)
        for generation_id, _validation in valid:
            if generation_id not in retained:
                retained.append(generation_id)
            if len(retained) >= self.retention:
                break
        return retained

    def _cleanup_staging_unlocked(self) -> None:
        for path in self.staging_path.iterdir():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
        _fsync_directory(self.staging_path)

    def _abandoned_staging_ids_unlocked(self) -> list[str]:
        if not self.staging_path.is_dir():
            return []
        return sorted(
            (
                path.name
                if self._valid_generation_id(path.name)
                else "unrecognized-staging-directory"
            )
            for path in self.staging_path.iterdir()
            if path.is_dir() and not path.is_symlink()
        )
