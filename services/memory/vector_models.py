from __future__ import annotations

import errno
import hashlib
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

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


def _empty_matrix(dimensions: int) -> np.ndarray:
    return np.empty((0, dimensions), dtype=np.float32)


def _matrix(rows: list[np.ndarray], dimensions: int) -> np.ndarray:
    if not rows:
        return _empty_matrix(dimensions)
    matrix = np.stack(rows).astype(np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != dimensions:
        raise ValueError("Vector dimensions do not match the embedding provider.")
    return matrix
