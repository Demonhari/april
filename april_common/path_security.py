from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path

from april_common.errors import PermissionDeniedError, ValidationError

SENSITIVE_SEGMENTS = {
    ".ssh",
    "Keychains",
    "Library/Keychains",
    "Library/Application Support/Google/Chrome",
    "Library/Application Support/Firefox",
    ".aws",
    ".azure",
    ".config/gcloud",
    "/etc",
    "/private/etc",
}

MODEL_SUFFIXES = {".gguf", ".bin", ".safetensors", ".onnx"}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".css",
    ".html",
    ".sql",
    ".csv",
    ".log",
}


@dataclass(frozen=True, slots=True)
class PathPolicy:
    allowed_roots: tuple[Path, ...]
    max_read_bytes: int
    max_write_bytes: int


def _contains_null(path: str | Path) -> bool:
    return "\x00" in str(path)


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            break
        current = current.parent
    return current


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _deny_sensitive(path: Path) -> None:
    parts = set(path.parts)
    for segment in SENSITIVE_SEGMENTS:
        if segment.startswith("/"):
            if _is_relative_to(path, Path(segment)):
                raise PermissionDeniedError("Access to sensitive system paths is denied.")
        elif "/" in segment:
            segment_path = Path(segment)
            for index in range(len(path.parts)):
                if Path(*path.parts[index : index + len(segment_path.parts)]) == segment_path:
                    raise PermissionDeniedError("Access to sensitive credential paths is denied.")
        elif segment in parts:
            raise PermissionDeniedError("Access to sensitive credential paths is denied.")


def normalize_existing_path(path: str | Path, policy: PathPolicy) -> Path:
    if _contains_null(path):
        raise ValidationError("Path contains a null byte.")
    requested = Path(path).expanduser()
    resolved = requested.resolve(strict=True)
    _deny_sensitive(resolved)
    roots = tuple(root.expanduser().resolve() for root in policy.allowed_roots)
    if not any(_is_relative_to(resolved, root) for root in roots):
        raise PermissionDeniedError("Path is outside configured allowed roots.")
    return resolved


def normalize_new_path(path: str | Path, policy: PathPolicy) -> Path:
    if _contains_null(path):
        raise ValidationError("Path contains a null byte.")
    requested = Path(path).expanduser()
    parent = _nearest_existing_parent(requested)
    resolved_parent = parent.resolve(strict=True)
    _deny_sensitive(resolved_parent)
    roots = tuple(root.expanduser().resolve() for root in policy.allowed_roots)
    if not any(_is_relative_to(resolved_parent, root) for root in roots):
        raise PermissionDeniedError("Path is outside configured allowed roots.")
    if requested.is_absolute():
        return resolved_parent / requested.relative_to(parent)
    return (resolved_parent / requested.relative_to(parent)).resolve()


def ensure_text_file(path: Path, *, max_bytes: int) -> None:
    if path.suffix.lower() in MODEL_SUFFIXES:
        raise PermissionDeniedError("Model and binary artifact files are not returned as text.")
    size = path.stat().st_size
    if size > max_bytes:
        raise PermissionDeniedError("File exceeds configured maximum read size.", {"size": size})
    sample = path.read_bytes()[:4096]
    if b"\x00" in sample:
        raise PermissionDeniedError("Binary files are not returned as text.")
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return
    content_type, _ = mimetypes.guess_type(path.name)
    if content_type and not content_type.startswith("text/"):
        raise PermissionDeniedError("Unsupported non-text file type.")
