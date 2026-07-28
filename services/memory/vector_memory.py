from __future__ import annotations

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

if TYPE_CHECKING:
    from april_common.audit import AuditLogger

FORMAT_VERSION = 3
DEFAULT_RETENTION = 2
DEFAULT_EMBED_BATCH_SIZE = 32
MAX_EMBED_BATCH_SIZE = 256
_GENERATION_RE = re.compile(r"^g-\d{8}T\d{12}Z-[0-9a-f]{12}$")
_REQUIRED_FILES = ("records.json", "vectors.npy", "metadata.json")
_DIRECTORY_FSYNC_UNSUPPORTED = {
    errno.EINVAL,
    errno.ENOTSUP,
    getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
    errno.EBADF,
}
_PROCESS_LOCKS: dict[Path, threading.RLock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class GenerationValidationResult:
    ok: bool
    generation_id: str | None
    failure_reasons: tuple[str, ...] = ()
    record_count: int = 0
    vector_count: int = 0
    dimensions: int | None = None
    provider: str | None = None
    hashes_valid: bool = False
    compatible: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _LoadedIndex:
    records: list[dict[str, Any]]
    vectors: np.ndarray
    generation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    storage_mode: str = "empty"
    active_generation: str | None = None
    fallback_active: bool = False
    failure_reasons: list[str] = field(default_factory=list)


def _process_lock(path: Path) -> threading.RLock:
    with _PROCESS_LOCKS_GUARD:
        return _PROCESS_LOCKS.setdefault(path, threading.RLock())


def _is_beneath(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    """Durably record directory entries where the platform supports it."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError as exc:
        if exc.errno in _DIRECTORY_FSYNC_UNSUPPORTED:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in _DIRECTORY_FSYNC_UNSUPPORTED:
                raise
    finally:
        os.close(descriptor)


class VectorMemory:
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

    def upsert(
        self,
        *,
        record_id: str,
        content: str,
        metadata: VectorMetadata,
    ) -> None:
        self.upsert_many([(record_id, content, metadata)])

    def upsert_many(self, items: list[tuple[str, str, VectorMetadata]]) -> None:
        if not items:
            return
        with self._locked():
            loaded = self._load_effective_unlocked(require_compatible=True)
            records = list(loaded.records)
            rows = [loaded.vectors[index].copy() for index in range(len(records))]
            by_id = {str(record["id"]): index for index, record in enumerate(records)}
            embedded = self._embed_texts([content for _record_id, content, _metadata in items])
            for item_index, (record_id, content, metadata) in enumerate(items):
                record = {
                    "id": record_id,
                    "content": content,
                    "metadata": metadata.model_dump(),
                }
                existing = by_id.get(record_id)
                if existing is None:
                    by_id[record_id] = len(records)
                    records.append(record)
                    rows.append(embedded[item_index])
                else:
                    records[existing] = record
                    rows[existing] = embedded[item_index]
            self._publish_unlocked(
                records,
                _matrix(rows, self._configured_dimensions()),
                source=loaded,
            )

    def delete(self, record_id: str) -> bool:
        with self._locked():
            loaded = self._load_effective_unlocked(require_compatible=True)
            kept = [
                index for index, record in enumerate(loaded.records) if record["id"] != record_id
            ]
            if len(kept) == len(loaded.records):
                return False
            self._publish_unlocked(
                [loaded.records[index] for index in kept],
                loaded.vectors[kept] if kept else _empty_matrix(self._configured_dimensions()),
                source=loaded,
            )
            return True

    def delete_stale_for_path(
        self,
        path: str,
        valid_content_hashes: set[str],
        *,
        source_type: str | None = None,
        source_id: str | None = None,
        project_id: str | None = None,
    ) -> int:
        with self._locked():
            loaded = self._load_effective_unlocked(require_compatible=True)
            kept: list[int] = []
            removed = 0
            for index, record in enumerate(loaded.records):
                metadata = record["metadata"]
                scoped = metadata.get("path") == path
                if source_type is not None:
                    scoped = scoped and metadata.get("source_type") == source_type
                if source_id is not None:
                    scoped = scoped and metadata.get("source_id") == source_id
                if project_id is not None:
                    scoped = scoped and metadata.get("project_id") == project_id
                if scoped and metadata.get("content_hash") not in valid_content_hashes:
                    removed += 1
                else:
                    kept.append(index)
            if removed:
                self._publish_unlocked(
                    [loaded.records[index] for index in kept],
                    loaded.vectors[kept] if kept else _empty_matrix(self._configured_dimensions()),
                    source=loaded,
                )
            return removed

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        project_id: str | None = None,
        source_type: str | None = None,
    ) -> list[SearchResult]:
        query_vector = self.embedding.embed(query).astype(np.float32)
        loaded = self._read_index()
        if not loaded.records:
            return []
        scores = loaded.vectors @ query_vector
        results: list[SearchResult] = []
        for index, record in enumerate(loaded.records):
            if project_id is not None and record["metadata"].get("project_id") != project_id:
                continue
            if source_type is not None and record["metadata"].get("source_type") != source_type:
                continue
            results.append(
                SearchResult(
                    id=record["id"],
                    score=float(scores[index]),
                    content=record["content"],
                    metadata=record["metadata"],
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    def sources(self, *, source_type: str) -> list[dict[str, Any]]:
        loaded = self._read_index()
        by_source: dict[str, dict[str, Any]] = {}
        for record in loaded.records:
            metadata = record["metadata"]
            if metadata.get("source_type") != source_type:
                continue
            source_id = str(metadata.get("source_id"))
            entry = by_source.setdefault(
                source_id, {"source_id": source_id, "paths": set(), "chunk_count": 0}
            )
            entry["chunk_count"] += 1
            if metadata.get("path"):
                entry["paths"].add(str(metadata["path"]))
        return [
            {
                "source_id": entry["source_id"],
                "paths": sorted(entry["paths"]),
                "chunk_count": entry["chunk_count"],
            }
            for entry in sorted(by_source.values(), key=lambda item: item["source_id"])
        ]

    def index_chunks(
        self,
        *,
        source_type: str,
        source_id: str,
        chunks: list[tuple[str, str, int | None, int | None]],
        project_id: str | None = None,
    ) -> None:
        paths = {chunk_path for chunk_path, _, _, _ in chunks}
        items: list[tuple[str, str, VectorMetadata]] = []
        for path, content, start_line, end_line in chunks:
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            item_metadata = VectorMetadata(
                source_type=source_type,
                source_id=source_id,
                project_id=project_id,
                path=path,
                start_line=start_line,
                end_line=end_line,
                content_hash=content_hash,
                created_at=utc_now_iso(),
            )
            record_id = hashlib.sha256(
                f"{source_type}:{source_id}:{path}:{content_hash}".encode()
            ).hexdigest()
            items.append((record_id, content, item_metadata))

        with self._locked():
            loaded = self._load_effective_unlocked(require_compatible=True)
            current_ids = {record_id for record_id, _content, _metadata in items}
            kept = []
            for index, record in enumerate(loaded.records):
                record_metadata = record["metadata"]
                same_scope = (
                    record_metadata.get("source_type") == source_type
                    and record_metadata.get("source_id") == source_id
                    and record_metadata.get("project_id") == project_id
                )
                if same_scope and (
                    record_metadata.get("path") not in paths or record["id"] not in current_ids
                ):
                    continue
                kept.append(index)
            records = [loaded.records[index] for index in kept]
            rows = [loaded.vectors[index].copy() for index in kept]
            by_id = {str(record["id"]): index for index, record in enumerate(records)}
            embedded = self._embed_texts([item[1] for item in items])
            for item_index, (record_id, content, item_metadata) in enumerate(items):
                record = {
                    "id": record_id,
                    "content": content,
                    "metadata": item_metadata.model_dump(),
                }
                existing = by_id.get(record_id)
                if existing is None:
                    by_id[record_id] = len(records)
                    records.append(record)
                    rows.append(embedded[item_index])
                else:
                    records[existing] = record
                    rows[existing] = embedded[item_index]
            self._publish_unlocked(
                records,
                _matrix(rows, self._configured_dimensions()),
                source=loaded,
            )

    def reset(self) -> None:
        with self._locked():
            loaded = self._load_effective_unlocked(require_compatible=False)
            self._publish_unlocked(
                [],
                _empty_matrix(self._configured_dimensions()),
                source=loaded,
            )

    def reindex(self, *, progress: Callable[[int, int], None] | None = None) -> int:
        """Re-embed every record and atomically publish exactly one generation."""
        with self._locked():
            loaded = self._load_effective_unlocked(require_compatible=False)
            records = list(loaded.records)
            loaded.vectors = np.empty((0, 0), dtype=np.float32)
            vectors = self._embed_texts(
                [str(record["content"]) for record in records],
                progress=progress,
            )
            completed_at = utc_now_iso()
            self._publish_unlocked(
                records,
                vectors,
                source=loaded,
                last_successful_reindex_at=completed_at,
            )
            return len(records)

    def rebuild_memory_namespace(
        self,
        items: list[tuple[str, str, VectorMetadata]],
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> int:
        with self._locked():
            loaded = self._load_effective_unlocked(require_compatible=False)
            preserved = [
                record
                for record in loaded.records
                if record.get("metadata", {}).get("source_type") != "memory"
            ]
            loaded.vectors = np.empty((0, 0), dtype=np.float32)
            records = [
                *preserved,
                *[
                    {
                        "id": record_id,
                        "content": content,
                        "metadata": metadata.model_dump(),
                    }
                    for record_id, content, metadata in items
                ],
            ]
            vectors = self._embed_texts(
                [str(record["content"]) for record in records],
                progress=progress,
            )
            completed_at = utc_now_iso()
            self._publish_unlocked(
                records,
                vectors,
                source=loaded,
                last_successful_reindex_at=completed_at,
            )
            return len(items)

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
        self.audit.write(
            {
                "event": "memory.vector_index_recovery",
                "generation_id": generation_id,
                "reason_code": reason,
            }
        )

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


def _empty_matrix(dimensions: int) -> np.ndarray:
    return np.empty((0, dimensions), dtype=np.float32)


def _matrix(rows: list[np.ndarray], dimensions: int) -> np.ndarray:
    if not rows:
        return _empty_matrix(dimensions)
    matrix = np.stack(rows).astype(np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != dimensions:
        raise ValueError("Vector dimensions do not match the embedding provider.")
    return matrix
