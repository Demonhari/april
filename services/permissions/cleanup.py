"""Scoped, two-stage log/cache cleanup.

This deliberately reuses the patch-artifact security model (content-addressed
immutable artifact, approval-bound metadata, fail-closed revalidation, one-time
use, mutation lock) rather than inventing a weaker second mechanism. There is no
generic or recursive delete here: cleanup can only ever touch ordinary files
beneath an APRIL-owned root (``logs`` or the audio cache), enumerated at plan
time into an immutable manifest that the Level-4 apply step is bound to.

Stage 1 (``plan_log_cleanup``, Level 1, read-only) enumerates candidates and
writes an immutable manifest. Stage 2 (``apply_log_cleanup``, Level 4) requires
exact approval bound to that manifest, revalidates every file's identity, deletes
only the exact candidate set, and marks the manifest one-time-use.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import stat
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from april_common.errors import PermissionDeniedError, ValidationError
from april_common.project_scope import sha256_bytes, sha256_file
from april_common.settings import AprilSettings, get_settings
from april_common.time import utc_now, utc_now_iso
from services.memory.schemas import ApprovalRecord
from skills.schemas import ToolResult

CLEANUP_MANIFEST_VERSION = 1
MANIFEST_ID_RE = re.compile(r"^[a-f0-9]{64}$")

CleanupTarget = Literal["logs", "audio_cache"]
ALLOWED_TARGETS: tuple[CleanupTarget, ...] = ("logs", "audio_cache")

# Never enumerated as deletable, even when old: the audit trail and the
# directory-structure marker must survive cleanup.
PROTECTED_BASENAMES = frozenset({".gitkeep", "audit.jsonl"})

# Conservative defaults; configurable via configs/tools.yaml -> tools.log_cleanup.
DEFAULT_MAX_CANDIDATE_FILES = 1000
DEFAULT_MAX_TOTAL_BYTES = 1_073_741_824  # 1 GiB

_CLEANUP_LOCKS: dict[str, asyncio.Lock] = {}


# --- limits / policy -------------------------------------------------------


class CleanupLimits:
    def __init__(
        self,
        *,
        allowed_targets: tuple[CleanupTarget, ...] = ALLOWED_TARGETS,
        max_candidate_files: int = DEFAULT_MAX_CANDIDATE_FILES,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
        min_age_days: int = 0,
    ) -> None:
        self.allowed_targets = allowed_targets
        self.max_candidate_files = max_candidate_files
        self.max_total_bytes = max_total_bytes
        self.min_age_days = min_age_days


def load_cleanup_limits(settings: AprilSettings | None = None) -> CleanupLimits:
    # Imported lazily so this module stays importable from skills without a cycle.
    from april_common.effective_config import load_tools_file

    active = settings or get_settings()
    try:
        policy = load_tools_file(active.home).tools.log_cleanup
    except Exception:
        return CleanupLimits()
    return CleanupLimits(
        allowed_targets=tuple(policy.allowed_targets),
        max_candidate_files=policy.max_candidate_files,
        max_total_bytes=policy.max_total_bytes,
        min_age_days=policy.min_age_days,
    )


# --- root resolution -------------------------------------------------------


def resolve_cleanup_root(target: str, settings: AprilSettings | None = None) -> Path:
    """Map a controlled target enum to its APRIL-owned root directory.

    The root is always derived from trusted settings; a caller-supplied path is
    never accepted, which is why ``plan_log_cleanup`` exposes no root argument.
    """
    active = settings or get_settings()
    if target == "logs":
        return active.logs_path.resolve()
    if target == "audio_cache":
        return active.audio_cache_path.resolve()
    raise PermissionDeniedError(
        "Unknown cleanup target.", {"target": target, "allowed": list(ALLOWED_TARGETS)}
    )


# --- enumeration -----------------------------------------------------------


def enumerate_candidates(
    *,
    target: str,
    older_than_days: int,
    settings: AprilSettings | None = None,
    limits: CleanupLimits | None = None,
) -> tuple[Path, list[dict[str, Any]], int]:
    active = settings or get_settings()
    active_limits = limits or load_cleanup_limits(active)
    if target not in active_limits.allowed_targets:
        raise PermissionDeniedError(
            "Cleanup target is not allowed.",
            {"target": target, "allowed": list(active_limits.allowed_targets)},
        )
    if older_than_days < 0:
        raise ValidationError("older_than_days must be >= 0.")
    effective_age = max(int(older_than_days), active_limits.min_age_days)
    root = resolve_cleanup_root(target, active)
    candidates: list[dict[str, Any]] = []
    total_bytes = 0
    if not root.is_dir():
        return root, candidates, 0
    cutoff = utc_now().timestamp() - effective_age * 86400
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Do not descend into symlinked directories.
        dirnames[:] = [name for name in dirnames if not (Path(dirpath) / name).is_symlink()]
        for filename in sorted(filenames):
            if filename in PROTECTED_BASENAMES:
                continue
            path = Path(dirpath) / filename
            try:
                info = path.lstat()
            except OSError:
                continue
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                continue
            if effective_age > 0 and info.st_mtime > cutoff:
                continue
            relpath = path.relative_to(root).as_posix()
            candidates.append(
                {
                    "relpath": relpath,
                    "size": int(info.st_size),
                    "sha256": sha256_file(path),
                    "mtime": _iso_from_timestamp(info.st_mtime),
                }
            )
            total_bytes += int(info.st_size)
            if len(candidates) > active_limits.max_candidate_files:
                raise PermissionDeniedError(
                    "Cleanup candidate count exceeds the configured maximum.",
                    {"max_candidate_files": active_limits.max_candidate_files},
                )
            if total_bytes > active_limits.max_total_bytes:
                raise PermissionDeniedError(
                    "Cleanup candidate total size exceeds the configured maximum.",
                    {"max_total_bytes": active_limits.max_total_bytes},
                )
    candidates.sort(key=lambda item: item["relpath"])
    return root, candidates, total_bytes


# --- manifest store (content-addressed, immutable) -------------------------


def build_cleanup_manifest(
    *,
    target: str,
    older_than_days: int,
    settings: AprilSettings | None = None,
    limits: CleanupLimits | None = None,
) -> dict[str, Any]:
    active = settings or get_settings()
    root, candidates, total_bytes = enumerate_candidates(
        target=target,
        older_than_days=older_than_days,
        settings=active,
        limits=limits,
    )
    manifest = {
        "manifest_version": CLEANUP_MANIFEST_VERSION,
        "target": target,
        "root": str(root),
        "older_than_days": int(older_than_days),
        "created_at": utc_now_iso(),
        "candidate_count": len(candidates),
        "total_bytes": total_bytes,
        "candidates": candidates,
    }
    stored = store_cleanup_manifest(manifest)
    return {
        "manifest_id": stored["manifest_id"],
        "manifest_sha256": stored["manifest_id"],
        "path": stored["path"],
        "target": target,
        "root": str(root),
        "candidate_count": len(candidates),
        "total_bytes": total_bytes,
        "relative_paths": [item["relpath"] for item in candidates],
    }


def _canonical_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def store_cleanup_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    canonical = _canonical_manifest_bytes(manifest)
    manifest_id = sha256_bytes(canonical)
    directory = _cleanup_artifact_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{manifest_id}.json"
    if path.exists():
        if sha256_file(path) != manifest_id:
            raise PermissionDeniedError("Cleanup manifest digest mismatch in artifact store.")
        return {"manifest_id": manifest_id, "path": str(path)}
    temporary = directory / f".{manifest_id}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"manifest_id": manifest_id, "path": str(path)}


def load_cleanup_manifest(manifest_id: str) -> dict[str, Any]:
    if not MANIFEST_ID_RE.fullmatch(manifest_id):
        raise PermissionDeniedError("Invalid cleanup manifest ID.")
    path = _cleanup_artifact_dir() / f"{manifest_id}.json"
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise PermissionDeniedError("Cleanup manifest is missing.") from exc
    if sha256_bytes(raw) != manifest_id:
        # Content-addressed: any edit to the manifest fails closed here.
        raise PermissionDeniedError("Cleanup manifest digest mismatch (tampered).")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise PermissionDeniedError("Cleanup manifest is malformed.")
    return data


# --- one-time-use marker ---------------------------------------------------


def _consumed_marker(manifest_id: str) -> Path:
    return _cleanup_artifact_dir() / f"{manifest_id}.consumed"


def is_manifest_consumed(manifest_id: str) -> bool:
    return _consumed_marker(manifest_id).exists()


def mark_manifest_consumed(manifest_id: str) -> None:
    marker = _consumed_marker(manifest_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(utc_now_iso(), encoding="utf-8")


# --- approval metadata + verification --------------------------------------


async def build_log_cleanup_approval_metadata(
    *, manifest_id: str, expected_side_effects: list[str]
) -> dict[str, Any]:
    manifest = load_cleanup_manifest(manifest_id)
    return {
        "artifact_type": "log_cleanup",
        "artifact_version": CLEANUP_MANIFEST_VERSION,
        "manifest_id": manifest_id,
        "manifest_sha256": manifest_id,
        "target": manifest.get("target"),
        "root": manifest.get("root"),
        "candidate_count": manifest.get("candidate_count"),
        "total_bytes": manifest.get("total_bytes"),
        "expected_side_effects": expected_side_effects,
    }


def _verify_manifest(record: ApprovalRecord) -> dict[str, Any] | ToolResult:
    metadata = record.metadata
    manifest_id = str(metadata.get("manifest_id", ""))
    if metadata.get("artifact_type") != "log_cleanup" or not manifest_id:
        return _failed("Cleanup approval is missing immutable manifest metadata.")
    args_manifest = str(record.args.get("manifest_id", ""))
    if args_manifest != manifest_id:
        return _failed("Cleanup arguments do not match the approved manifest.")
    if is_manifest_consumed(manifest_id):
        return _failed("Cleanup manifest has already been used.")
    try:
        manifest = load_cleanup_manifest(manifest_id)
    except PermissionDeniedError as exc:
        return _failed(exc.message, details=exc.details)
    # Root containment: the manifest root must still equal the configured,
    # APRIL-owned root for its declared target. A caller cannot redirect cleanup.
    target = str(manifest.get("target", ""))
    try:
        expected_root = resolve_cleanup_root(target)
    except PermissionDeniedError as exc:
        return _failed(exc.message, details=exc.details)
    if str(manifest.get("root")) != str(expected_root):
        return _failed("Cleanup manifest root is outside the configured APRIL-owned root.")
    return manifest


async def verify_log_cleanup_approval(record: ApprovalRecord) -> ToolResult | None:
    result = _verify_manifest(record)
    if isinstance(result, ToolResult):
        return result
    return None


async def apply_approved_log_cleanup(record: ApprovalRecord) -> ToolResult:
    manifest_or_failure = _verify_manifest(record)
    if isinstance(manifest_or_failure, ToolResult):
        return manifest_or_failure
    manifest = manifest_or_failure
    manifest_id = str(record.metadata.get("manifest_id", ""))
    root = Path(str(manifest.get("root"))).resolve()
    async with _cleanup_lock(str(root)):
        if is_manifest_consumed(manifest_id):
            return _failed("Cleanup manifest has already been used.")
        deleted: list[str] = []
        skipped: list[dict[str, str]] = []
        deleted_bytes = 0
        for entry in manifest.get("candidates", []):
            relpath = str(entry.get("relpath", ""))
            outcome = _delete_one(root, relpath, entry)
            if outcome is None:
                deleted.append(relpath)
                deleted_bytes += int(entry.get("size", 0))
            else:
                skipped.append({"relpath": relpath, "reason": outcome})
        # One-time use: mark consumed even if some files were already gone so the
        # manifest cannot be replayed for a second approval.
        mark_manifest_consumed(manifest_id)
    return ToolResult(
        ok=True,
        stdout=f"Deleted {len(deleted)} file(s), skipped {len(skipped)}.",
        data={
            "manifest_id": manifest_id,
            "target": manifest.get("target"),
            "deleted_count": len(deleted),
            "deleted_bytes": deleted_bytes,
            "skipped_count": len(skipped),
            "deleted_relative_paths": deleted,
            "skipped": skipped,
        },
        risk_level="system_action",
        permission_level=4,
    )


def _delete_one(root: Path, relpath: str, entry: dict[str, Any]) -> str | None:
    """Delete one candidate after revalidation. Returns a skip reason or None."""
    rel = Path(relpath)
    if rel.is_absolute() or ".." in rel.parts:
        return "path traversal rejected"
    candidate = root / rel
    try:
        info = candidate.lstat()
    except OSError:
        return "missing"
    if stat.S_ISLNK(info.st_mode):
        return "symlink rejected"
    if not stat.S_ISREG(info.st_mode):
        return "not a regular file"
    # Guard a parent directory swapped for a symlink after planning.
    resolved = candidate.resolve()
    if root != resolved and root not in resolved.parents:
        return "outside configured root"
    if int(info.st_size) != int(entry.get("size", -1)):
        return "size changed after planning"
    try:
        if sha256_file(candidate) != str(entry.get("sha256")):
            return "content changed after planning"
    except OSError:
        return "unreadable"
    try:
        candidate.unlink()
    except OSError as exc:
        return f"unlink failed: {exc.__class__.__name__}"
    return None


# --- helpers ---------------------------------------------------------------


@asynccontextmanager
async def _cleanup_lock(root: str) -> AsyncIterator[None]:
    lock = _CLEANUP_LOCKS.get(root)
    if lock is None:
        lock = asyncio.Lock()
        _CLEANUP_LOCKS[root] = lock
    async with lock:
        yield


def _cleanup_artifact_dir() -> Path:
    return get_settings().resolve_path(Path("data/artifacts/cleanup"))


def _iso_from_timestamp(timestamp: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat().replace("+00:00", "Z")


def _failed(message: str, *, details: dict[str, Any] | None = None) -> ToolResult:
    return ToolResult(
        ok=False,
        stderr=message,
        data=details or {},
        risk_level="system_action",
        permission_level=4,
    )
