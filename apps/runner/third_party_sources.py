"""Validation and local build helpers for APRIL's third-party source manifest.

This module intentionally does not fetch, install, import, or execute third-party
source.  A source becomes buildable only after an operator stages it under the
declared repository path and records a full upstream revision plus the upstream
license and notice files in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

MANIFEST_RELATIVE_PATH = Path("third_party/source-manifest.json")
COLIBRI_ENTRY_ID = "colibri"
_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_ENTRY_KINDS = {"runtime", "reference"}
_ENTRY_STATUSES = {"vendored", "metadata_only"}


class ThirdPartySourceError(ValueError):
    """Raised when third-party source metadata is unsafe or incomplete."""


@dataclass(frozen=True)
class ThirdPartySource:
    id: str
    kind: Literal["runtime", "reference"]
    status: Literal["vendored", "metadata_only"]
    upstream_name: str
    upstream_repository: str | None
    revision: str | None
    source_path: Path
    license_files: tuple[Path, ...]
    notice_files: tuple[Path, ...]
    adaptation_metadata: Path | None
    production_dependency: bool
    fingerprint_scope: str


@dataclass(frozen=True)
class ThirdPartyManifest:
    schema_version: str
    manifest_id: str
    entries: tuple[ThirdPartySource, ...]

    @property
    def colibri(self) -> ThirdPartySource:
        for entry in self.entries:
            if entry.id == COLIBRI_ENTRY_ID:
                return entry
        raise ThirdPartySourceError("Manifest does not define the Colibri runtime entry.")


def manifest_path(root: Path) -> Path:
    return root / MANIFEST_RELATIVE_PATH


def load_source_manifest(
    root: Path,
    path: Path | None = None,
) -> ThirdPartyManifest:
    """Load and validate the checked-in manifest without touching source trees."""

    manifest_file = path or manifest_path(root)
    try:
        raw = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ThirdPartySourceError("Third-party source manifest is unreadable.") from exc
    if not isinstance(raw, dict):
        raise ThirdPartySourceError("Third-party source manifest must be an object.")
    schema_version = raw.get("schema_version")
    manifest_id = raw.get("manifest_id")
    entries = raw.get("entries")
    if not isinstance(schema_version, str) or not schema_version:
        raise ThirdPartySourceError("Third-party source manifest has no schema version.")
    if not isinstance(manifest_id, str) or not manifest_id:
        raise ThirdPartySourceError("Third-party source manifest has no manifest ID.")
    if not isinstance(entries, list) or not entries:
        raise ThirdPartySourceError("Third-party source manifest has no entries.")

    parsed: list[ThirdPartySource] = []
    seen: set[str] = set()
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise ThirdPartySourceError("Third-party source manifest contains an invalid entry.")
        entry = _parse_entry(root, raw_entry)
        if entry.id in seen:
            raise ThirdPartySourceError(f"Duplicate third-party source ID: {entry.id}")
        seen.add(entry.id)
        parsed.append(entry)
    if COLIBRI_ENTRY_ID not in seen:
        raise ThirdPartySourceError("Third-party source manifest must define Colibri.")
    return ThirdPartyManifest(schema_version, manifest_id, tuple(parsed))


def manifest_identity_digest(root: Path, path: Path | None = None) -> str:
    """Return a digest of canonical manifest data, not local source contents."""

    manifest_file = path or manifest_path(root)
    try:
        raw = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ThirdPartySourceError("Third-party source manifest is unreadable.") from exc
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def inspect_source_manifest(root: Path, path: Path | None = None) -> dict[str, Any]:
    """Return a redacted, deterministic doctor payload for the source layout."""

    try:
        manifest = load_source_manifest(root, path)
    except ThirdPartySourceError as exc:
        return {
            "ready": False,
            "manifest_valid": False,
            "reason": str(exc),
        }

    entries: list[dict[str, Any]] = []
    for entry in manifest.entries:
        source_present = _safe_is_dir(root, entry.source_path)
        license_present = bool(entry.license_files) and all(
            _safe_is_file(root, license_path) for license_path in entry.license_files
        )
        notice_present = bool(entry.notice_files) and all(
            _safe_is_file(root, notice_path) for notice_path in entry.notice_files
        )
        adaptation_present = entry.adaptation_metadata is None or _safe_is_file(
            root, entry.adaptation_metadata
        )
        pinned = entry.revision is not None and bool(_REVISION_RE.fullmatch(entry.revision))
        source_ready = _entry_ready(
            entry,
            source_present=source_present,
            pinned=pinned,
            license_present=license_present,
            notice_present=notice_present,
            adaptation_present=adaptation_present,
        )
        reason = "ready" if source_ready else _source_reason(entry, source_present, pinned)
        entries.append(
            {
                "id": entry.id,
                "kind": entry.kind,
                "status": entry.status,
                "source_path": entry.source_path.as_posix(),
                "revision_pinned": pinned,
                "license_notice_present": license_present and notice_present,
                "adaptation_metadata_present": adaptation_present,
                "production_dependency": entry.production_dependency,
                "ready": source_ready,
                "reason": reason,
            }
        )

    all_ready = all(bool(entry["ready"]) for entry in entries)
    return {
        "ready": all_ready,
        "manifest_valid": True,
        "manifest_id": manifest.manifest_id,
        "schema_version": manifest.schema_version,
        "manifest_digest": manifest_identity_digest(root, path),
        "entries": entries,
        "production_package_includes_third_party": False,
    }


def build_colibri(
    root: Path,
    *,
    jobs: int = 1,
    path: Path | None = None,
) -> dict[str, Any]:
    """Build staged Colibri source with a fixed, local-only CMake invocation.

    The command never clones source, resolves dependencies, starts a service, or
    handles model weights.  The build directory is intentionally ignored by Git.
    """

    if jobs < 1 or jobs > 64:
        raise ThirdPartySourceError("Build parallelism must be between 1 and 64.")
    manifest = load_source_manifest(root, path)
    entry = manifest.colibri
    report = inspect_source_manifest(root, path)
    colibri_report = next(item for item in report.get("entries", []) if item["id"] == entry.id)
    if not colibri_report["ready"]:
        raise ThirdPartySourceError(
            "Colibri source is not staged with a pinned revision and preserved license/notice."
        )
    source = _safe_relative_path(root, entry.source_path)
    if not (source / "CMakeLists.txt").is_file():
        raise ThirdPartySourceError("Staged Colibri source has no CMakeLists.txt build entrypoint.")
    cmake = shutil.which("cmake")
    if cmake is None:
        raise ThirdPartySourceError("CMake is required to build staged Colibri source.")
    build_dir = _safe_relative_path(root, Path("third_party/colibri/build"))
    build_dir.mkdir(parents=True, exist_ok=True)
    common = [
        cmake,
        "-S",
        str(source),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
    ]
    _run_local_build(common, cwd=root)
    _run_local_build([cmake, "--build", str(build_dir), "--parallel", str(jobs)], cwd=root)
    return {
        "built": True,
        "source": entry.source_path.as_posix(),
        "build_directory": "third_party/colibri/build",
        "jobs": jobs,
        "network_access": False,
        "model_weights": "not handled",
    }


def _parse_entry(root: Path, raw: dict[str, Any]) -> ThirdPartySource:
    entry_id = raw.get("id")
    kind = raw.get("kind")
    status = raw.get("status")
    upstream = raw.get("upstream")
    source_path = raw.get("source_path")
    if (
        not isinstance(entry_id, str)
        or not entry_id
        or kind not in _ENTRY_KINDS
        or status not in _ENTRY_STATUSES
        or not isinstance(upstream, dict)
        or not isinstance(source_path, str)
    ):
        raise ThirdPartySourceError("Third-party source manifest contains an incomplete entry.")
    if not isinstance(upstream.get("name"), str) or not upstream["name"]:
        raise ThirdPartySourceError(f"Source {entry_id} has no upstream name.")
    if upstream.get("repository") is not None and not isinstance(upstream.get("repository"), str):
        raise ThirdPartySourceError(f"Source {entry_id} has an invalid repository URL.")
    revision = upstream.get("revision")
    if revision is not None and (
        not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision)
    ):
        raise ThirdPartySourceError(f"Source {entry_id} revision is not a full Git SHA-1.")
    source = _manifest_path(root, source_path, entry_id)
    license_files = _manifest_paths(root, raw.get("license_files", []), entry_id)
    notice_files = _manifest_paths(root, raw.get("notice_files", []), entry_id)
    adaptation_raw = raw.get("adaptation_metadata")
    adaptation = (
        _manifest_path(root, adaptation_raw, entry_id) if adaptation_raw is not None else None
    )
    production_dependency = raw.get("production_dependency", False)
    if not isinstance(production_dependency, bool):
        raise ThirdPartySourceError(f"Source {entry_id} has an invalid dependency flag.")
    fingerprint_scope = raw.get("fingerprint_scope")
    if not isinstance(fingerprint_scope, str) or not fingerprint_scope:
        raise ThirdPartySourceError(f"Source {entry_id} has no fingerprint scope.")
    if kind == "reference" and production_dependency:
        raise ThirdPartySourceError(
            f"Reference source {entry_id} cannot be a production dependency."
        )
    return ThirdPartySource(
        id=entry_id,
        kind=kind,
        status=status,
        upstream_name=upstream["name"],
        upstream_repository=upstream.get("repository"),
        revision=revision,
        source_path=source,
        license_files=license_files,
        notice_files=notice_files,
        adaptation_metadata=adaptation,
        production_dependency=production_dependency,
        fingerprint_scope=fingerprint_scope,
    )


def _manifest_paths(root: Path, value: Any, entry_id: str) -> tuple[Path, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ThirdPartySourceError(f"Source {entry_id} has invalid license or notice paths.")
    return tuple(_manifest_path(root, item, entry_id) for item in value)


def _manifest_path(root: Path, value: Any, entry_id: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ThirdPartySourceError(f"Source {entry_id} has an invalid relative path.")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ThirdPartySourceError(f"Source {entry_id} contains a path outside the repository.")
    try:
        candidate.relative_to(Path("third_party"))
    except ValueError as exc:
        raise ThirdPartySourceError(
            f"Source {entry_id} paths must remain under third_party."
        ) from exc
    return candidate


def _safe_relative_path(root: Path, relative: Path) -> Path:
    root_resolved = root.resolve()
    candidate = (root / relative).resolve(strict=False)
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ThirdPartySourceError("Third-party path escapes the repository.") from exc
    return candidate


def _safe_is_dir(root: Path, relative: Path) -> bool:
    try:
        return _safe_relative_path(root, relative).is_dir()
    except ThirdPartySourceError:
        return False


def _safe_is_file(root: Path, relative: Path) -> bool:
    try:
        return _safe_relative_path(root, relative).is_file()
    except ThirdPartySourceError:
        return False


def _source_reason(entry: ThirdPartySource, source_present: bool, pinned: bool) -> str:
    if entry.status != "vendored":
        return "provenance_only" if entry.kind == "reference" else "source_not_staged"
    if not source_present:
        return "source_directory_missing"
    if not pinned:
        return "revision_not_pinned"
    return "license_or_notice_metadata_incomplete"


def _entry_ready(
    entry: ThirdPartySource,
    *,
    source_present: bool,
    pinned: bool,
    license_present: bool,
    notice_present: bool,
    adaptation_present: bool,
) -> bool:
    if entry.kind == "reference" and entry.status == "metadata_only":
        return True
    return (
        entry.status == "vendored"
        and source_present
        and pinned
        and bool(entry.license_files)
        and license_present
        and bool(entry.notice_files)
        and notice_present
        and adaptation_present
    )


def _run_local_build(argv: list[str], *, cwd: Path) -> None:
    try:
        subprocess.run(argv, check=True, cwd=cwd)
    except OSError as exc:
        raise ThirdPartySourceError("Colibri local build could not start.") from exc
    except subprocess.CalledProcessError as exc:
        raise ThirdPartySourceError(
            f"Colibri local build failed with exit code {exc.returncode}."
        ) from exc
