from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from april_common.time import parse_utc_iso, utc_now
from services.evolution.write_guard import EvolutionWriteGuard
from services.memory.sqlite_memory import SqliteMemory

# Deterministic decay policy for machine-written memories. User-requested
# memories never decay automatically — the user asked for them, so only the
# user (or an explicit supersede) retires them.
UNUSED_AFTER_DAYS = 14
DECAY_FACTOR = 0.9
FADE_CONFIDENCE_FLOOR = 0.3
FADE_GRACE_DAYS = 30


@dataclass(slots=True)
class MemoryDecayReport:
    decayed: int = 0
    faded: int = 0
    details: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "memories_decayed": self.decayed,
            "memories_fading": self.faded,
            "decay_details": self.details[:50],
        }


async def apply_memory_decay(
    memory: SqliteMemory,
    *,
    guard: EvolutionWriteGuard,
    now: datetime | None = None,
    unused_after_days: int = UNUSED_AFTER_DAYS,
    decay_factor: float = DECAY_FACTOR,
    fade_confidence_floor: float = FADE_CONFIDENCE_FLOOR,
    fade_grace_days: int = FADE_GRACE_DAYS,
) -> MemoryDecayReport:
    """Fade stale machine-written memories without ever deleting a row.

    A machine memory unused for ``unused_after_days`` loses confidence by
    ``decay_factor`` per (at most nightly) run. When confidence drops below
    ``fade_confidence_floor`` the memory starts *fading*: ``expires_at`` is set
    ``fade_grace_days`` ahead. After that timestamp passes the row stops being
    served (expired) but stays on disk, inspectable via memory inspect.
    """
    guard.validate_table("memories")
    current = now or utc_now()
    threshold = current - timedelta(days=unused_after_days)
    report = MemoryDecayReport()
    for record in await memory.list_memories():
        if record.source == "user":
            continue
        last_activity = record.last_used_at or record.created_at
        try:
            last_dt = parse_utc_iso(last_activity)
        except ValueError:
            continue
        if last_dt > threshold:
            continue
        new_confidence = round(record.confidence * decay_factor, 4)
        expires_at: str | None = record.expires_at
        fading = False
        if new_confidence < fade_confidence_floor and record.expires_at is None:
            expiry = current + timedelta(days=fade_grace_days)
            expires_at = expiry.isoformat().replace("+00:00", "Z")
            fading = True
        await memory.set_memory_decay(
            record.id, confidence=new_confidence, expires_at=expires_at
        )
        report.decayed += 1
        if fading:
            report.faded += 1
            report.details.append(
                f"memory {record.id} fading (confidence {new_confidence}), "
                f"expires {expires_at}"
            )
        else:
            report.details.append(
                f"memory {record.id} decayed to confidence {new_confidence}"
            )
    return report
