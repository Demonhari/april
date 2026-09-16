"""Validation and local build helpers for APRIL's third-party source manifest.

This module intentionally does not fetch, install, or import third-party source.
It executes only explicit, allowlisted local build and diagnostic commands after
an operator has verified the declared source, revision, license, and notices.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

MANIFEST_RELATIVE_PATH = Path("third_party/source-manifest.json")
COLIBRI_ENTRY_ID = "colibri"
_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_DIGEST_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_ENTRY_KINDS = {"runtime", "reference"}
_ENTRY_STATUSES = {"vendored", "metadata_only"}
_FORBIDDEN_DIRECTORY_NAMES = {
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "build",
    "cache",
    "checkpoints",
    "dist",
    "models",
    "node_modules",
    "weights",
}
_FORBIDDEN_SUFFIXES = {
    ".a",
    ".bin",
    ".dylib",
    ".gguf",
    ".onnx",
    ".o",
    ".safetensors",
    ".so",
}
_FORBIDDEN_FILE_NAMES = {".coli_ssd", ".coli_usage"}
_COLIBRI_MIN_PYTHON = (3, 10)
_COLIBRI_PYTHON_NAMES = ("python3.14", "python3.13", "python3.12", "python3.11", "python3")
_COLIBRI_PYTHON_PATHS = (
    "/opt/homebrew/bin/python3.14",
    "/usr/local/bin/python3.14",
    "/opt/homebrew/opt/python@3.14/bin/python3.14",
    "/usr/local/opt/python@3.14/bin/python3.14",
)


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
    original_upstream_snapshot_digest: str | None
    current_april_vendor_digest: str | None
    snapshot_exclusions: tuple[str, ...]


@dataclass(frozen=True)
class ThirdPartyManifest:
    schema_version: str
    manifest_id: str
    provenance_status: str
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
    provenance_status = raw.get("provenance_status")
    entries = raw.get("entries")
    if not isinstance(schema_version, str) or not schema_version:
        raise ThirdPartySourceError("Third-party source manifest has no schema version.")
    if not isinstance(manifest_id, str) or not manifest_id:
        raise ThirdPartySourceError("Third-party source manifest has no manifest ID.")
    if not isinstance(provenance_status, str) or not provenance_status:
        raise ThirdPartySourceError("Third-party source manifest has no provenance status.")
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
    return ThirdPartyManifest(schema_version, manifest_id, provenance_status, tuple(parsed))


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
        # An upstream project may have no NOTICE file.  An empty manifest list
        # therefore means "no notice required", while every listed notice must
        # still be present.
        notice_present = all(_safe_is_file(root, notice_path) for notice_path in entry.notice_files)
        adaptation_path = entry.adaptation_metadata
        adaptation_present = adaptation_path is not None and _safe_is_file(root, adaptation_path)
        pinned = entry.revision is not None and bool(_REVISION_RE.fullmatch(entry.revision))
        snapshot_digest, nested_git, forbidden_paths = _vendor_tree_details(
            root, entry.source_path, entry.snapshot_exclusions
        )
        snapshot_matches = (
            snapshot_digest is not None
            and entry.current_april_vendor_digest is not None
            and snapshot_digest == entry.current_april_vendor_digest
        )
        source_ready = _entry_ready(
            entry,
            source_present=source_present,
            pinned=pinned,
            license_present=license_present,
            notice_present=notice_present,
            adaptation_present=adaptation_present,
            snapshot_matches=snapshot_matches,
            nested_git=nested_git,
            forbidden_paths=forbidden_paths,
        )
        reason = (
            "ready"
            if source_ready
            else _source_reason(
                entry,
                source_present,
                pinned,
                license_present,
                notice_present,
                adaptation_present,
                snapshot_matches,
                nested_git,
                forbidden_paths,
            )
        )
        entries.append(
            {
                "id": entry.id,
                "kind": entry.kind,
                "status": entry.status,
                "source_path": entry.source_path.as_posix(),
                "revision_pinned": pinned,
                "license_present": license_present,
                "notice_present": notice_present,
                "license_notice_present": license_present and notice_present,
                "notice_required": bool(entry.notice_files),
                "adaptation_metadata_present": adaptation_present,
                "snapshot_digest": snapshot_digest,
                "snapshot_matches": snapshot_matches,
                "nested_git": nested_git,
                "forbidden_paths": forbidden_paths,
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
        "provenance_status": manifest.provenance_status,
        "manifest_digest": manifest_identity_digest(root, path),
        "entries": entries,
        "production_package_includes_third_party": False,
    }


def build_colibri(
    root: Path,
    *,
    engine: str = "qwen36",
    arch: str = "native",
    path: Path | None = None,
) -> dict[str, Any]:
    """Build staged Colibri source with Colibri's fixed local Makefile target.

    The command never clones source, resolves dependencies, starts a service, or
    handles model weights.  It passes only an allowlisted engine and architecture.
    """

    if engine != "qwen36":
        raise ThirdPartySourceError("Only the reviewed qwen36 Colibri engine is allowed.")
    if arch != "native":
        raise ThirdPartySourceError("Only the reviewed native Colibri architecture is allowed.")
    manifest = load_source_manifest(root, path)
    entry = manifest.colibri
    report = inspect_source_manifest(root, path)
    colibri_report = next(item for item in report.get("entries", []) if item["id"] == entry.id)
    if not colibri_report["ready"]:
        raise ThirdPartySourceError(
            "Colibri source is not ready with a pinned revision and preserved license/notices."
        )
    source = _safe_relative_path(root, entry.source_path)
    makefile_dir = source / "c"
    if not makefile_dir.is_dir() or not (makefile_dir / "Makefile").is_file():
        raise ThirdPartySourceError("Staged Colibri source has no c/Makefile build entrypoint.")
    if shutil.which("make") is None:
        raise ThirdPartySourceError("Make is required to build staged Colibri source.")
    _run_local_build(
        ["make", "-C", str(makefile_dir), engine, f"ARCH={arch}"],
        cwd=root,
    )
    return {
        "built": True,
        "source": entry.source_path.as_posix(),
        "engine": engine,
        "arch": arch,
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
    original_digest = raw.get("original_upstream_snapshot_digest")
    current_digest = raw.get("current_april_vendor_digest")
    for field_name, value in (
        ("original_upstream_snapshot_digest", original_digest),
        ("current_april_vendor_digest", current_digest),
    ):
        if value is not None and (not isinstance(value, str) or not _DIGEST_RE.fullmatch(value)):
            raise ThirdPartySourceError(f"Source {entry_id} has an invalid {field_name}.")
    if status == "vendored" and (original_digest is None or current_digest is None):
        raise ThirdPartySourceError(f"Vendored source {entry_id} has no snapshot digest.")
    exclusions = raw.get("snapshot_exclusions", [])
    if not isinstance(exclusions, list) or not all(
        isinstance(item, str)
        and item
        and not Path(item).is_absolute()
        and ".." not in Path(item).parts
        for item in exclusions
    ):
        raise ThirdPartySourceError(f"Source {entry_id} has invalid snapshot exclusions.")
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
        original_upstream_snapshot_digest=original_digest,
        current_april_vendor_digest=current_digest,
        snapshot_exclusions=tuple(exclusions),
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


def _source_reason(
    entry: ThirdPartySource,
    source_present: bool,
    pinned: bool,
    license_present: bool,
    notice_present: bool,
    adaptation_present: bool,
    snapshot_matches: bool,
    nested_git: bool,
    forbidden_paths: list[str],
) -> str:
    if entry.status != "vendored":
        return "provenance_only" if entry.kind == "reference" else "source_not_staged"
    if not source_present:
        return "source_directory_missing"
    if not pinned:
        return "revision_not_pinned"
    if not license_present:
        return "license_missing"
    if not notice_present:
        return "listed_notice_missing"
    if not adaptation_present:
        return "adaptation_metadata_missing"
    if nested_git:
        return "nested_git_directory_present"
    if forbidden_paths:
        return "generated_or_model_artifact_present"
    if not snapshot_matches:
        return "snapshot_digest_mismatch"
    return "license_or_notice_metadata_incomplete"


def _entry_ready(
    entry: ThirdPartySource,
    *,
    source_present: bool,
    pinned: bool,
    license_present: bool,
    notice_present: bool,
    adaptation_present: bool,
    snapshot_matches: bool,
    nested_git: bool,
    forbidden_paths: list[str],
) -> bool:
    return (
        entry.status == "vendored"
        and source_present
        and pinned
        and bool(entry.license_files)
        and license_present
        and notice_present
        and bool(entry.adaptation_metadata)
        and adaptation_present
        and snapshot_matches
        and not nested_git
        and not forbidden_paths
    )


def _vendor_tree_details(
    root: Path,
    relative: Path,
    exclusions: tuple[str, ...] = (),
) -> tuple[str | None, bool, list[str]]:
    """Inspect only the vendored tree and return relative, redacted findings."""

    try:
        source = _safe_relative_path(root, relative)
    except ThirdPartySourceError:
        return None, False, [relative.as_posix()]
    if not source.is_dir():
        return None, False, []
    records: list[dict[str, str]] = []
    nested_git = False
    forbidden: list[str] = []
    for candidate in sorted(source.rglob("*")):
        candidate_relative = candidate.relative_to(source)
        parts = candidate_relative.parts
        display_path = candidate_relative.as_posix()
        if ".git" in parts:
            nested_git = True
            continue
        if (
            any(part in _FORBIDDEN_DIRECTORY_NAMES for part in parts)
            or candidate.name == "qwen36"
            or candidate.name in _FORBIDDEN_FILE_NAMES
        ):
            if len(forbidden) < 20:
                forbidden.append(display_path)
            continue
        if candidate.is_symlink():
            if len(forbidden) < 20:
                forbidden.append(display_path)
            continue
        if not candidate.is_file():
            continue
        if display_path == "APRIL_ADAPTATION.md" or display_path in exclusions:
            continue
        if candidate.suffix.lower() in _FORBIDDEN_SUFFIXES:
            if len(forbidden) < 20:
                forbidden.append(display_path)
            continue
        records.append({"path": display_path, "sha256": _file_digest(candidate)})
    material = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(material).hexdigest(), nested_git, forbidden


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vendor_snapshot_digest(
    root: Path, relative: Path, exclusions: tuple[str, ...] = ()
) -> str | None:
    """Return the deterministic digest used for a vendored source tree."""

    digest, _nested_git, _forbidden_paths = _vendor_tree_details(root, relative, exclusions)
    return digest


def resolve_colibri_python(candidates: list[str] | None = None) -> Path:
    """Resolve a Python interpreter compatible with Colibri's launcher."""

    if candidates is None:
        candidates = []
        override = os.environ.get("APRIL_COLIBRI_PYTHON")
        if override:
            candidates.append(override)
        candidates.append(sys.executable)
        candidates.extend(name for name in _COLIBRI_PYTHON_NAMES if shutil.which(name))
        candidates.extend(path for path in _COLIBRI_PYTHON_PATHS if Path(path).exists())

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            if candidate == sys.executable:
                version = sys.version_info[:2]
            else:
                result = subprocess.run(
                    [candidate, "-c", "import sys; print(sys.version_info[0:2])"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                match = re.search(r"\((\d+),\s*(\d+)\)", result.stdout)
                if match is None:
                    continue
                version = (int(match.group(1)), int(match.group(2)))
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
        if version >= _COLIBRI_MIN_PYTHON:
            return Path(candidate)
    raise ThirdPartySourceError(
        "Colibri requires Python 3.10 or newer; set APRIL_COLIBRI_PYTHON to a "
        "compatible interpreter."
    )


def run_colibri_command(
    root: Path,
    subcommand: str,
    *,
    model: Path,
    options: list[str] | None = None,
) -> int:
    """Run a reviewed Colibri launcher command without starting it indirectly."""

    if subcommand not in {"info", "doctor", "plan", "serve"}:
        raise ThirdPartySourceError("Unsupported Colibri launcher command.")
    report = inspect_source_manifest(root)
    colibri = next(
        (entry for entry in report.get("entries", []) if entry.get("id") == COLIBRI_ENTRY_ID),
        None,
    )
    if not report.get("ready") or not colibri or not colibri.get("ready"):
        raise ThirdPartySourceError(
            "Colibri source is not ready according to the third-party doctor."
        )
    script = (
        _safe_relative_path(root, load_source_manifest(root).colibri.source_path) / "c" / "coli"
    )
    if not script.is_file():
        raise ThirdPartySourceError("The vendored Colibri launcher c/coli is missing.")
    interpreter = resolve_colibri_python()
    argv = [str(interpreter), str(script), subcommand, "--model", str(model)]
    argv.extend(options or [])
    try:
        result = subprocess.run(argv, cwd=root, check=False)
    except OSError as exc:
        raise ThirdPartySourceError("Colibri launcher could not start.") from exc
    return result.returncode


def _run_local_build(argv: list[str], *, cwd: Path) -> None:
    try:
        subprocess.run(argv, check=True, cwd=cwd)
    except OSError as exc:
        raise ThirdPartySourceError("Colibri local build could not start.") from exc
    except subprocess.CalledProcessError as exc:
        raise ThirdPartySourceError(
            f"Colibri local build failed with exit code {exc.returncode}."
        ) from exc
