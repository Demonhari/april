"""Deterministic repository identity and state-bound verification evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RepositoryState:
    project_id: str
    head_commit: str | None
    tracked_diff_digest: str
    untracked_digest: str
    affected_paths: tuple[str, ...] = ()

    @property
    def digest(self) -> str:
        return _digest(self.as_identity())

    def as_identity(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "head_commit": self.head_commit,
            "tracked_diff_digest": self.tracked_diff_digest,
            "untracked_digest": self.untracked_digest,
            "affected_paths": list(self.affected_paths),
        }

    @classmethod
    def capture(cls, root: Path, *, project_id: str | None = None) -> RepositoryState:
        resolved = root.expanduser().resolve(strict=False)
        identity = project_id or _digest(str(resolved))[:24]
        head = _git(resolved, ["rev-parse", "--verify", "HEAD"])
        diff = _git(resolved, ["diff", "--no-ext-diff", "--binary"])
        tracked_paths = _git(resolved, ["diff", "--name-only", "--no-ext-diff", "-z"])
        untracked = _git(resolved, ["ls-files", "--others", "--exclude-standard", "-z"])
        paths = tuple(
            sorted({path for path in (*tracked_paths.split("\0"), *untracked.split("\0")) if path})
        )
        untracked_material: list[dict[str, str]] = []
        for relative in paths:
            candidate = (resolved / relative).resolve(strict=False)
            if not _within(candidate, resolved) or not candidate.is_file():
                untracked_material.append({"path": relative, "digest": "unreadable"})
                continue
            untracked_material.append({"path": relative, "digest": _file_digest(candidate)})
        return cls(
            project_id=identity,
            head_commit=head or None,
            tracked_diff_digest=_digest(diff),
            untracked_digest=_digest(untracked_material),
            affected_paths=paths,
        )


@dataclass(frozen=True, slots=True)
class VerificationEvidence:
    command_id: str
    exit_status: int
    result_status: str
    stdout_digest: str
    stderr_digest: str
    truncated: bool
    repository_state_digest: str
    test_targets: tuple[str, ...] = ()
    summary: str = ""
    observed_at: str | None = None

    def is_current(self, state: RepositoryState) -> bool:
        return self.repository_state_digest == state.digest


VerificationSubject = RepositoryState


def _git(root: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout if result.returncode == 0 else ""


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return "unreadable"
    return digest.hexdigest()


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True
