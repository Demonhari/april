"""Small, side-effect-free checks for operator-supplied GGUF artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

GgufArtifactStatus = Literal[
    "valid",
    "missing",
    "not_regular_file",
    "unreadable",
    "invalid_gguf_header",
]


def gguf_artifact_status(path: Path) -> GgufArtifactStatus:
    """Return a redaction-safe structural status without loading the model."""
    if not path.exists():
        return "missing"
    if not path.is_file():
        return "not_regular_file"
    try:
        with path.open("rb") as handle:
            magic = handle.read(4)
    except OSError:
        return "unreadable"
    return "valid" if magic == b"GGUF" else "invalid_gguf_header"
