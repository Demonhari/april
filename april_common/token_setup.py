from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GeneratedTokens:
    api_token: str
    runtime_token: str


def generate_tokens() -> GeneratedTokens:
    return GeneratedTokens(
        api_token=secrets.token_urlsafe(32),
        runtime_token=secrets.token_urlsafe(32),
    )


def write_token_env_file(path: Path, tokens: GeneratedTokens) -> None:
    path = path.expanduser()
    existing: list[str] = []
    if path.exists():
        existing = path.read_text(encoding="utf-8").splitlines()

    replacements = {
        "APRIL_API_TOKEN": tokens.api_token,
        "APRIL_RUNTIME_TOKEN": tokens.runtime_token,
    }
    written: set[str] = set()
    lines: list[str] = []
    for line in existing:
        key, separator, _value = line.partition("=")
        if separator and key in replacements:
            lines.append(f"{key}={replacements[key]}")
            written.add(key)
        else:
            lines.append(line)
    for key, value in replacements.items():
        if key not in written:
            lines.append(f"{key}={value}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Some filesystems do not support POSIX modes. The caller can still use the file.
        return
