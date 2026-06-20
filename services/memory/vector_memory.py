from __future__ import annotations

import hashlib
import json
import shutil
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
        self.records_path = self.path / "records.jsonl"
        self.path.mkdir(parents=True, exist_ok=True)

    def health(self) -> dict[str, Any]:
        return {
            "ok": self.path.exists(),
            "path": str(self.path),
            "embedding": "hashed-token",
            "dimensions": self.embedding.dimensions,
        }

    def upsert(
        self,
        *,
        record_id: str,
        content: str,
        metadata: VectorMetadata,
    ) -> None:
        records = [record for record in self._read_records() if record["id"] != record_id]
        records.append(
            {
                "id": record_id,
                "content": content,
                "metadata": metadata.model_dump(),
                "vector": self.embedding.embed(content).tolist(),
            }
        )
        self._write_records(records)

    def delete(self, record_id: str) -> bool:
        records = self._read_records()
        updated = [record for record in records if record["id"] != record_id]
        self._write_records(updated)
        return len(updated) != len(records)

    def delete_stale_for_path(self, path: str, valid_content_hashes: set[str]) -> int:
        records = self._read_records()
        updated = [
            record
            for record in records
            if not (
                record["metadata"].get("path") == path
                and record["metadata"].get("content_hash") not in valid_content_hashes
            )
        ]
        self._write_records(updated)
        return len(records) - len(updated)

    def search(
        self, query: str, *, limit: int = 10, project_id: str | None = None
    ) -> list[SearchResult]:
        query_vector = self.embedding.embed(query)
        results: list[SearchResult] = []
        for record in self._read_records():
            if project_id is not None and record["metadata"].get("project_id") != project_id:
                continue
            vector = np.asarray(record["vector"], dtype=np.float32)
            score = float(np.dot(query_vector, vector))
            results.append(
                SearchResult(
                    id=record["id"],
                    score=score,
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
        valid_hashes: set[str] = set()
        paths = {chunk_path for chunk_path, _, _, _ in chunks}
        for path, content, start_line, end_line in chunks:
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            valid_hashes.add(content_hash)
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
            self.upsert(record_id=record_id, content=content, metadata=metadata)
        for path in paths:
            self.delete_stale_for_path(path, valid_hashes)

    def reset(self) -> None:
        if self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _read_records(self) -> list[dict[str, Any]]:
        if not self.records_path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.records_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    def _write_records(self, records: list[dict[str, Any]]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        with self.records_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        vectors = np.asarray([record["vector"] for record in records], dtype=np.float32)
        np.save(self.vectors_path, vectors)
