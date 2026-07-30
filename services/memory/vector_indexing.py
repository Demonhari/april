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


class VectorIndexing(VectorMemoryBase):
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
            self._cleanup_staging_unlocked()
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
            self._cleanup_staging_unlocked()
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
