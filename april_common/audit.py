from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from april_common.errors import AprilError
from april_common.time import utc_now_iso

SECRET_KEYWORDS = ("token", "secret", "password", "authorization", "credential", "api_key")


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if any(secret in key.lower() for secret in SECRET_KEYWORDS):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = redact(item)
        return redacted
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


class AuditLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, entry: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": utc_now_iso(),
            **redact(entry),
        }
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        except OSError as exc:
            raise AprilError("AUDIT_LOG_FAILED", "Unable to write audit log.", 500) from exc
