from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from april_common.path_security import MODEL_SUFFIXES, PathPolicy
from april_common.settings import get_settings


def current_path_policy() -> PathPolicy:
    settings = get_settings()
    return PathPolicy(
        allowed_roots=tuple(settings.allowed_roots),
        max_read_bytes=settings.paths.max_file_read_bytes,
        max_write_bytes=settings.paths.max_file_write_bytes,
    )


def read_gitignore_patterns(root: Path) -> list[str]:
    path = root / ".gitignore"
    if not path.exists():
        return []
    patterns: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            patterns.append(stripped)
    return patterns


def ignored(path: Path, *, root: Path, patterns: list[str]) -> bool:
    if ".git" in path.parts:
        return True
    if path.suffix.lower() in MODEL_SUFFIXES:
        return True
    rel = str(path.relative_to(root)) if path.is_relative_to(root) else path.name
    return any(
        fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern) for pattern in patterns
    )


def safe_regex(pattern: str) -> re.Pattern[str]:
    if len(pattern) > 200:
        raise ValueError("Search pattern is too long.")
    if re.search(r"(\.\*|\.\+|\[[^\]]+\]\*){2,}", pattern):
        raise ValueError("Potentially catastrophic regular expression is denied.")
    return re.compile(pattern, re.IGNORECASE)
