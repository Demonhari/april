from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from april_common.settings import AprilSettings
from april_common.time import utc_now_iso
from services.evolution.write_guard import EvolutionWriteGuard

WakeListeningState = Literal["idle", "listening", "muted"]


def wake_status_path(settings: AprilSettings) -> Path:
    return settings.evolution_path / "wake" / "status.json"


def write_wake_status(settings: AprilSettings, state: WakeListeningState) -> None:
    payload = {"schema_version": 1, "state": state, "updated_at": utc_now_iso()}
    EvolutionWriteGuard(settings).write_text(
        wake_status_path(settings),
        json.dumps(payload, sort_keys=True) + "\n",
    )


def read_wake_status(settings: AprilSettings) -> dict[str, str | int | None]:
    if settings.mute_flag_path.exists():
        return {"schema_version": 1, "state": "muted", "updated_at": None}
    try:
        payload = json.loads(wake_status_path(settings).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "state": "idle", "updated_at": None}
    if not isinstance(payload, dict):
        return {"schema_version": 1, "state": "idle", "updated_at": None}
    state = payload.get("state")
    if state not in {"idle", "listening", "muted"}:
        state = "idle"
    updated_at = payload.get("updated_at")
    return {
        "schema_version": 1,
        "state": str(state),
        "updated_at": str(updated_at) if isinstance(updated_at, str) else None,
    }
