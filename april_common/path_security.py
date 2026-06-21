from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path

from april_common.errors import PermissionDeniedError, ValidationError

SENSITIVE_SEGMENTS = {
    ".ssh",
    ".aws",
    ".azure",
    ".gnupg",
    ".config/gcloud",
    "Keychains",
    "Library/Keychains",
    "Library/Application Support/Google/Chrome",
    "Library/Application Support/Firefox",
    "/etc",
    "/private/etc",
}

SENSITIVE_FILENAMES = {
    ".env",
    ".netrc",
    ".npmrc",
    "credentials",
    "credentials.json",
    "credentials.yml",
    "credentials.yaml",
    "token",
    "tokens.json",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "known_hosts",
    "april.db",
}

SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}
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


def is_path_within_roots(path: Path, roots: list[Path] | tuple[Path, ...]) -> bool:
    resolved = path.expanduser().resolve(strict=False)
    resolved_roots = tuple(root.expanduser().resolve(strict=False) for root in roots)
    return any(_is_relative_to(resolved, root) for root in resolved_roots)


def deny_sensitive_path(path: Path) -> None:
    parts = set(path.parts)
    casefold_parts = {part.casefold() for part in path.parts}
    for segment in SENSITIVE_SEGMENTS:
        if segment.startswith("/"):
            if _is_relative_to(path, Path(segment)):
                raise PermissionDeniedError("Access to sensitive system paths is denied.")
        elif "/" in segment:
            segment_path = Path(segment)
            segment_casefold = tuple(part.casefold() for part in segment_path.parts)
            for index in range(len(path.parts)):
                if Path(*path.parts[index : index + len(segment_path.parts)]) == segment_path:
                    raise PermissionDeniedError("Access to sensitive credential paths is denied.")
                candidate = tuple(
                    part.casefold() for part in path.parts[index : index + len(segment_path.parts)]
                )
                if candidate == segment_casefold:
                    raise PermissionDeniedError("Access to sensitive credential paths is denied.")
        elif segment in parts or segment.casefold() in casefold_parts:
            raise PermissionDeniedError("Access to sensitive credential paths is denied.")
    name = path.name.casefold()
    if name in SENSITIVE_FILENAMES:
        raise PermissionDeniedError("Access to sensitive credential files is denied.")
    if name.startswith(".env."):
        raise PermissionDeniedError("Access to sensitive environment files is denied.")
    if path.suffix.casefold() in SENSITIVE_SUFFIXES:
        raise PermissionDeniedError("Access to private key files is denied.")
    lower_parts = [part.casefold() for part in path.parts]
    if len(lower_parts) >= 2 and lower_parts[-2:] == ["data", "april.db"]:
        raise PermissionDeniedError("Direct tool access to the APRIL database is denied.")


def normalize_existing_path(path: str | Path, policy: PathPolicy) -> Path:
    if _contains_null(path):
        raise ValidationError("Path contains a null byte.")
    requested = Path(path).expanduser()
    resolved = requested.resolve(strict=True)
    deny_sensitive_path(resolved)
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
    deny_sensitive_path(resolved_parent)
    roots = tuple(root.expanduser().resolve() for root in policy.allowed_roots)
    if not any(_is_relative_to(resolved_parent, root) for root in roots):
        raise PermissionDeniedError("Path is outside configured allowed roots.")
    if requested.is_absolute():
        resolved = resolved_parent / requested.relative_to(parent)
    else:
        resolved = (resolved_parent / requested.relative_to(parent)).resolve()
    deny_sensitive_path(resolved)
    return resolved


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
