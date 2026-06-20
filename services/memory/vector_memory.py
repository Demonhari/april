from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from april_common.time import utc_now_iso
from services.memory.embeddings import EmbeddingProvider, HashedTokenEmbedding
from services.memory.schemas import SearchResult, VectorMetadata


class VectorMemory:
    def __init__(self, path: Path, embedding: EmbeddingProvider | None = None) -> None:
        self.path = path
        self.embedding = embedding or HashedTokenEmbedding()
        self.vectors_path = self.path / "vectors.npy"
        self.records_json_path = self.path / "records.json"
        self.records_path = self.path / "records.jsonl"
        self.metadata_path = self.path / "metadata.json"
        self.lock_path = self.path / ".lock"
        self.path.mkdir(parents=True, exist_ok=True)

    def health(self) -> dict[str, Any]:
        records, _vectors = self._read_index()
        return {
            "ok": self.path.exists(),
            "path": str(self.path),
            "embedding": self.embedding.name,
            "dimensions": self.embedding.dimensions,
            "record_count": len(records),
        }

    def upsert(
        self,
        *,
        record_id: str,
        content: str,
        metadata: VectorMetadata,
    ) -> None:
        self.upsert_many([(record_id, content, metadata)])

    def upsert_many(self, items: list[tuple[str, str, VectorMetadata]]) -> None:
        with self._locked():
            records, vectors = self._read_index_unlocked()
            by_id = {record["id"]: index for index, record in enumerate(records)}
            vector_rows = [vectors[index] for index in range(len(records))]
            for record_id, content, metadata in items:
                record = {
                    "id": record_id,
                    "content": content,
                    "metadata": metadata.model_dump(),
                }
                vector = self.embedding.embed(content).astype(np.float32)
                existing = by_id.get(record_id)
                if existing is None:
                    by_id[record_id] = len(records)
                    records.append(record)
                    vector_rows.append(vector)
                else:
                    records[existing] = record
                    vector_rows[existing] = vector
            self._write_index_unlocked(records, _matrix(vector_rows, self.embedding.dimensions))

    def delete(self, record_id: str) -> bool:
        with self._locked():
            records, vectors = self._read_index_unlocked()
            kept_indexes = [
                index for index, record in enumerate(records) if record["id"] != record_id
            ]
            if len(kept_indexes) == len(records):
                return False
            self._write_index_unlocked(
                [records[index] for index in kept_indexes],
                vectors[kept_indexes]
                if len(kept_indexes)
                else _empty_matrix(self.embedding.dimensions),
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
            records, vectors = self._read_index_unlocked()
            kept_indexes: list[int] = []
            removed = 0
            for index, record in enumerate(records):
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
                    continue
                kept_indexes.append(index)
            if removed:
                self._write_index_unlocked(
                    [records[index] for index in kept_indexes],
                    vectors[kept_indexes]
                    if kept_indexes
                    else _empty_matrix(self.embedding.dimensions),
                )
            return removed

    def search(
        self, query: str, *, limit: int = 10, project_id: str | None = None
    ) -> list[SearchResult]:
        query_vector = self.embedding.embed(query).astype(np.float32)
        results: list[SearchResult] = []
        records, vectors = self._read_index()
        if not records:
            return []
        scores = vectors @ query_vector
        for index, record in enumerate(records):
            if project_id is not None and record["metadata"].get("project_id") != project_id:
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
            metadata = VectorMetadata(
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
            items.append((record_id, content, metadata))
        with self._locked():
            records, vectors = self._read_index_unlocked()
            current_ids = {record_id for record_id, _content, _metadata in items}
            kept_indexes: list[int] = []
            for index, record in enumerate(records):
                metadata = record["metadata"]
                same_scope = (
                    metadata.get("source_type") == source_type
                    and metadata.get("source_id") == source_id
                    and metadata.get("project_id") == project_id
                )
                if same_scope and (
                    metadata.get("path") not in paths or record["id"] not in current_ids
                ):
                    continue
                kept_indexes.append(index)
            kept_records = [records[index] for index in kept_indexes]
            kept_vectors = (
                vectors[kept_indexes] if kept_indexes else _empty_matrix(self.embedding.dimensions)
            )
            self._write_index_unlocked(kept_records, kept_vectors)
        self.upsert_many(items)

    def reset(self) -> None:
        if self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _read_index(self) -> tuple[list[dict[str, Any]], np.ndarray]:
        with self._locked():
            return self._read_index_unlocked()

    def _read_index_unlocked(self) -> tuple[list[dict[str, Any]], np.ndarray]:
        if self.records_json_path.exists() and self.vectors_path.exists():
            records = json.loads(self.records_json_path.read_text(encoding="utf-8"))
            vectors = np.load(self.vectors_path)
            if vectors.shape[0] != len(records):
                raise RuntimeError("Vector index record/vector count mismatch.")
            return records, vectors.astype(np.float32)
        if self.records_path.exists():
            legacy_records: list[dict[str, Any]] = []
            legacy_vectors: list[np.ndarray] = []
            for line in self.records_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                vector = np.asarray(record.pop("vector"), dtype=np.float32)
                legacy_records.append(record)
                legacy_vectors.append(vector)
            return legacy_records, _matrix(legacy_vectors, self.embedding.dimensions)
        return [], _empty_matrix(self.embedding.dimensions)

    def _write_index_unlocked(self, records: list[dict[str, Any]], vectors: np.ndarray) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": 2,
            "provider": self.embedding.name,
            "dimensions": self.embedding.dimensions,
            "record_count": len(records),
            "updated_at": utc_now_iso(),
            "failure_state": None,
        }
        self._replace_text(self.records_json_path, json.dumps(records, sort_keys=True))
        self._replace_npy(self.vectors_path, vectors.astype(np.float32))
        self._replace_text(self.metadata_path, json.dumps(metadata, sort_keys=True))

    def _replace_text(self, target: Path, content: str) -> None:
        fd, raw = tempfile.mkstemp(dir=self.path, prefix=f".{target.name}.", text=True)
        temp = Path(raw)
        try:
            with open(fd, "w", encoding="utf-8", closefd=True) as handle:
                handle.write(content)
                handle.write("\n")
            temp.replace(target)
        finally:
            if temp.exists():
                temp.unlink()

    def _replace_npy(self, target: Path, vectors: np.ndarray) -> None:
        fd, raw = tempfile.mkstemp(dir=self.path, prefix=f".{target.name}.", suffix=".npy")
        temp = Path(raw)
        try:
            with open(fd, "wb", closefd=True) as handle:
                np.save(handle, vectors)
            temp.replace(target)
        finally:
            if temp.exists():
                temp.unlink()

    @contextlib.contextmanager
    def _locked(self) -> Any:
        self.path.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _empty_matrix(dimensions: int) -> np.ndarray:
    return np.zeros((0, dimensions), dtype=np.float32)


def _matrix(vectors: list[np.ndarray], dimensions: int) -> np.ndarray:
    if not vectors:
        return _empty_matrix(dimensions)
    return np.asarray(vectors, dtype=np.float32).reshape((len(vectors), dimensions))
