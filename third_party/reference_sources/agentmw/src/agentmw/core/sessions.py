"""Session storage: persist agent traces as JSON files for later replay.

Sessions are stored as JSON files in `~/.agentmw/sessions/<id>.json` (or
`$AGENTMW_HOME/sessions/`). Chose filesystem over SQLite because traces are
immutable blobs — easier to diff, share, inspect with `cat`, and version-
control. SQLite stays for the reasoning library, where we need query.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

Message = dict[str, Any]


def _default_dir() -> Path:
    base = os.environ.get("AGENTMW_HOME") or os.path.expanduser("~/.agentmw")
    p = Path(base) / "sessions"
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class Session:
    id: str
    task: str
    messages: list[Message]
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_messages(cls, messages: list[Message], task: str | None = None) -> "Session":
        if task is None:
            for m in messages:
                if m.get("role") == "user":
                    c = m.get("content")
                    if isinstance(c, str):
                        task = c[:120]
                        break
                    if isinstance(c, list):
                        for b in c:
                            if isinstance(b, dict) and b.get("type") == "text":
                                task = b.get("text", "")[:120]
                                break
                    if task:
                        break
        return cls(
            id=uuid.uuid4().hex[:12],
            task=task or "(no task)",
            messages=messages,
        )


class SessionStore:
    def __init__(self, directory: Path | str | None = None) -> None:
        self.dir = Path(directory) if directory else _default_dir()
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.dir / f"{session_id}.json"

    def save(self, session: Session) -> Path:
        path = self._path(session.id)
        with open(path, "w") as f:
            json.dump(asdict(session), f, indent=2)
        return path

    def load(self, session_id: str) -> Session:
        path = self._path(session_id)
        if not path.is_file():
            # also try prefix match — short ids are convenient on CLI
            matches = list(self.dir.glob(f"{session_id}*.json"))
            if len(matches) == 1:
                path = matches[0]
            elif len(matches) > 1:
                raise ValueError(f"ambiguous session id '{session_id}'; matches {len(matches)}")
            else:
                raise FileNotFoundError(f"no session matches '{session_id}'")
        with open(path) as f:
            data = json.load(f)
        return Session(**data)

    def list(self) -> list[Session]:
        out: list[Session] = []
        for p in sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with open(p) as f:
                    out.append(Session(**json.load(f)))
            except (OSError, ValueError, TypeError):
                continue
        return out
