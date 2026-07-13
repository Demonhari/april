from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from april_common.audit import AuditLogger
from april_common.settings import AprilSettings
from april_common.time import utc_now_iso
from services.evolution.write_guard import EvolutionWriteGuard
from services.memory.database import Database

_STRUCTURAL_OVERLAY_RE = re.compile(
    r"(?im)^\s*(tools|permissions|allowed_tools|tool_registry|permission_level)\s*:"
)

# Header used when active overlay bytes are appended to an agent's effective
# system prompt. Overlays are advisory prose only: they can never change tools,
# permissions, configs, or policy, and repo prompt files are never modified.
LEARNED_GUIDANCE_HEADER = "## Learned guidance (local, advisory only)"

# Agents whose tools can modify the system (Forge writes code, Hand acts on the
# system). Their overlays never auto-apply, whatever the source: explicit user
# approval is always required.
WRITE_CAPABLE_AGENTS = frozenset({"coding_agent", "system_action_agent"})


@dataclass(frozen=True, slots=True)
class OverlayApplyResult:
    status: Literal["applied", "discarded", "approval_required", "pending_real_runtime"]
    agent: str
    version: int | None = None
    reason: str | None = None
    path: Path | None = None


@dataclass(frozen=True, slots=True)
class LadderThresholds:
    deep_confidence_threshold: float
    verified_confidence_threshold: float

    def to_payload(self) -> dict[str, float]:
        return {
            "deep_confidence_threshold": self.deep_confidence_threshold,
            "verified_confidence_threshold": self.verified_confidence_threshold,
        }


@dataclass(frozen=True, slots=True)
class LadderOverlayResult:
    status: Literal["applied", "discarded"]
    version: int | None = None
    reason: str | None = None
    path: Path | None = None
    thresholds: LadderThresholds | None = None


class PromptOverlayManager:
    def __init__(
        self,
        settings: AprilSettings,
        database: Database,
        *,
        audit: AuditLogger | None = None,
        guard: EvolutionWriteGuard | None = None,
    ) -> None:
        self.settings = settings
        self.database = database
        self.audit = audit
        self.guard = guard or EvolutionWriteGuard(settings, audit=audit)

    async def apply_candidate(
        self,
        *,
        agent: str,
        content: str,
        eval_score: float,
        baseline_score: float,
        source: Literal["dreamer", "forge", "hand"] = "dreamer",
        approved: bool = False,
    ) -> OverlayApplyResult:
        reason = self.rejection_reason(content)
        if reason is not None:
            self._audit("prompt_overlay_discarded", agent=agent, reason=reason)
            return OverlayApplyResult("discarded", agent, reason=reason)
        if eval_score < baseline_score:
            self._audit("prompt_overlay_discarded", agent=agent, reason="below baseline")
            return OverlayApplyResult("discarded", agent, reason="eval score below baseline")
        if (source in {"forge", "hand"} or agent in WRITE_CAPABLE_AGENTS) and not approved:
            self._audit(
                "prompt_overlay_approval_required",
                agent=agent,
                reason=f"source={source} agent={agent}",
            )
            return OverlayApplyResult("approval_required", agent, reason="user approval required")
        version = await self._next_version(agent)
        path = self.settings.evolution_path / "prompts" / agent / f"v{version}.overlay.txt"
        written = self.guard.write_text(path, content)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        self.guard.validate_table("prompt_versions")
        async with self.database.transaction() as conn:
            await conn.execute("UPDATE prompt_versions SET active = 0 WHERE agent = ?", (agent,))
            await conn.execute(
                """
                INSERT INTO prompt_versions(
                    id, agent, version, overlay_path, content_hash, active,
                    eval_score, baseline_score, created_at
                )
                VALUES(?, ?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (
                    f"{agent}:{version}",
                    agent,
                    version,
                    str(written),
                    content_hash,
                    eval_score,
                    baseline_score,
                    utc_now_iso(),
                ),
            )
        self._audit("prompt_overlay_applied", agent=agent, version=version)
        return OverlayApplyResult("applied", agent, version=version, path=written)

    async def active_overlay(self, agent: str) -> bytes | None:
        row = await self.database.fetchone(
            "SELECT overlay_path FROM prompt_versions WHERE agent = ? AND active = 1",
            (agent,),
        )
        if row is None:
            return None
        path = Path(str(row["overlay_path"]))
        if not path.exists():
            return None
        return path.read_bytes()

    async def active_overlay_text(self, agent: str) -> str | None:
        """Active overlay as bounded, policy-checked text for prompt assembly.

        Returns ``None`` when no overlay is active, its bytes are missing
        (e.g. data/evolution was deleted — stock behaviour), or the on-disk
        content was tampered into structural tool/permission changes. The text
        is re-bounded at read time so a hand-edited file cannot exceed the
        configured overlay budget.
        """
        raw = await self.active_overlay(agent)
        if raw is None:
            return None
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            return None
        max_chars = self.settings.evolution.prompt_overlay_max_chars
        if max_chars > 0:
            text = text[:max_chars]
        if _STRUCTURAL_OVERLAY_RE.search(text):
            self._audit(
                "prompt_overlay_blocked_at_load",
                agent=agent,
                reason="structural content in overlay bytes",
            )
            return None
        return text

    async def active_eval_score(self, agent: str) -> float | None:
        row = await self.database.fetchone(
            "SELECT eval_score FROM prompt_versions WHERE agent = ? AND active = 1",
            (agent,),
        )
        if row is None:
            return None
        try:
            return float(row["eval_score"])
        except (TypeError, ValueError):
            return None

    async def versions(self, *, agent: str | None = None) -> list[dict[str, object]]:
        if agent is None:
            rows = await self.database.fetchall(
                "SELECT * FROM prompt_versions ORDER BY agent, version"
            )
        else:
            rows = await self.database.fetchall(
                "SELECT * FROM prompt_versions WHERE agent = ? ORDER BY version",
                (agent,),
            )
        return [dict(row) for row in rows]

    async def rollback(self, *, agent: str, version: int) -> OverlayApplyResult:
        row = await self.database.fetchone(
            "SELECT * FROM prompt_versions WHERE agent = ? AND version = ?",
            (agent, version),
        )
        if row is None:
            return OverlayApplyResult("discarded", agent, reason="version not found")
        path = Path(str(row["overlay_path"]))
        if not path.exists():
            return OverlayApplyResult("discarded", agent, reason="overlay bytes missing")
        self.guard.validate_table("prompt_versions")
        async with self.database.transaction() as conn:
            await conn.execute("UPDATE prompt_versions SET active = 0 WHERE agent = ?", (agent,))
            await conn.execute(
                "UPDATE prompt_versions SET active = 1 WHERE agent = ? AND version = ?",
                (agent, version),
            )
        self._audit("prompt_overlay_rollback", agent=agent, version=version)
        return OverlayApplyResult("applied", agent, version=version, path=path)

    async def _next_version(self, agent: str) -> int:
        row = await self.database.fetchone(
            """
            SELECT COALESCE(MAX(version), 0) + 1 AS next_version
            FROM prompt_versions
            WHERE agent = ?
            """,
            (agent,),
        )
        return int(row["next_version"]) if row is not None else 1

    def rejection_reason(self, content: str) -> str | None:
        """Policy check shared by auto-apply and the user approval path."""
        if len(content) > self.settings.evolution.prompt_overlay_max_chars:
            return "overlay exceeds prompt_overlay_max_chars"
        if _STRUCTURAL_OVERLAY_RE.search(content):
            return "overlay attempts structural tool or permission changes"
        return None

    def _audit(
        self,
        event_type: str,
        *,
        agent: str,
        reason: str | None = None,
        version: int | None = None,
    ) -> None:
        if self.audit is not None:
            self.audit.write(
                {
                    "event_type": event_type,
                    "actor": "dreamer",
                    "agent": agent,
                    "reason": reason,
                    "version": version,
                }
            )


class LadderThresholdOverlayManager:
    """Versioned threshold overlays for the intelligence ladder.

    Overlay bytes are deliberately tiny and structural: exactly two float keys.
    Metadata and active version live in a separate pointer so loading an overlay
    can reject any extra key with zero effect.
    """

    def __init__(
        self,
        settings: AprilSettings,
        *,
        audit: AuditLogger | None = None,
        guard: EvolutionWriteGuard | None = None,
    ) -> None:
        self.settings = settings
        self.audit = audit
        self.guard = guard or EvolutionWriteGuard(settings, audit=audit)

    def active_thresholds(self) -> LadderThresholds:
        return _active_ladder_thresholds(self.settings, audit=self.audit)

    def apply_candidate(
        self,
        thresholds: LadderThresholds,
        *,
        eval_score: float,
        baseline_score: float,
    ) -> LadderOverlayResult:
        normalized = validate_ladder_threshold_overlay(thresholds.to_payload())
        if eval_score < baseline_score:
            self._audit("ladder_threshold_overlay_discarded", reason="below baseline")
            return LadderOverlayResult("discarded", reason="routing eval score below baseline")
        version = self._next_version()
        path = _ladder_overlay_path(self.settings, version)
        self.guard.write_text(
            path,
            json.dumps(normalized.to_payload(), indent=2, sort_keys=True) + "\n",
        )
        pointer = {
            "schema_version": 1,
            "active_version": version,
            "created_at": utc_now_iso(),
            "eval_score": eval_score,
            "baseline_score": baseline_score,
        }
        self.guard.write_text(
            _ladder_pointer_path(self.settings),
            json.dumps(pointer, indent=2, sort_keys=True) + "\n",
        )
        self._audit("ladder_threshold_overlay_applied", version=version)
        return LadderOverlayResult(
            "applied",
            version=version,
            path=path,
            thresholds=normalized,
        )

    def rollback(self, version: int | None = None) -> LadderOverlayResult:
        active = _read_ladder_pointer(self.settings)
        active_version = _active_ladder_version(active)
        target = version if version is not None else self._previous_version(active_version)
        if target is None:
            return LadderOverlayResult("discarded", reason="no prior threshold version")
        path = _ladder_overlay_path(self.settings, target)
        try:
            thresholds = validate_ladder_threshold_overlay(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return LadderOverlayResult("discarded", reason=f"threshold overlay invalid: {exc}")
        pointer = {
            "schema_version": 1,
            "active_version": target,
            "created_at": utc_now_iso(),
            "rollback_from": active_version,
        }
        self.guard.write_text(
            _ladder_pointer_path(self.settings),
            json.dumps(pointer, indent=2, sort_keys=True) + "\n",
        )
        self._audit("ladder_threshold_overlay_rollback", version=target)
        return LadderOverlayResult(
            "applied",
            version=target,
            path=path,
            thresholds=thresholds,
        )

    def _next_version(self) -> int:
        versions = [
            _parse_ladder_version(path)
            for path in (_ladder_config_dir(self.settings)).glob("ladder-v*.json")
        ]
        return max((version for version in versions if version is not None), default=0) + 1

    def _previous_version(self, active_version: int | None) -> int | None:
        versions = [
            version
            for version in (
                _parse_ladder_version(path)
                for path in (_ladder_config_dir(self.settings)).glob("ladder-v*.json")
            )
            if version is not None
        ]
        if active_version is None:
            return versions[-1] if versions else None
        prior = [version for version in sorted(versions) if version < active_version]
        return prior[-1] if prior else None

    def _audit(
        self,
        event_type: str,
        *,
        version: int | None = None,
        reason: str | None = None,
    ) -> None:
        if self.audit is not None:
            self.audit.write(
                {
                    "event_type": event_type,
                    "actor": "dreamer",
                    "version": version,
                    "reason": reason,
                }
            )


async def propose_ladder_thresholds_from_memory(
    settings: AprilSettings,
    memory: Any,
) -> LadderThresholds | None:
    """Deterministic nightly threshold nudge from persisted outcomes + feedback."""
    current = active_ladder_thresholds(settings)
    joined = await memory.database.fetchall(
        """
        SELECT runs.metadata_json, events.rating
        FROM feedback_events AS events
        JOIN agent_runs AS runs ON runs.id = events.agent_run_id
        ORDER BY events.created_at DESC
        LIMIT 100
        """
    )
    bad = 0
    good = 0
    for row in joined:
        try:
            metadata = json.loads(str(row["metadata_json"] or "{}"))
        except json.JSONDecodeError:
            continue
        rung = metadata.get("intelligence_rung") if isinstance(metadata, dict) else None
        if not isinstance(rung, int):
            continue
        if row["rating"] == "bad":
            bad += 1
        elif row["rating"] == "good":
            good += 1
    if bad == good:
        return None
    nudge = 0.05 if bad > good else -0.05
    return bounded_ladder_threshold_nudge(
        LadderThresholds(
            deep_confidence_threshold=current["deep_confidence_threshold"],
            verified_confidence_threshold=current["verified_confidence_threshold"],
        ),
        nudge=nudge,
    )


def bounded_ladder_threshold_nudge(
    current: LadderThresholds,
    *,
    nudge: float,
) -> LadderThresholds:
    bounded = max(-0.05, min(0.05, nudge))
    deep = max(0.2, min(0.6, current.deep_confidence_threshold + bounded))
    verified = max(0.5, min(0.9, current.verified_confidence_threshold + bounded))
    if deep >= verified:
        deep = min(deep, verified - 0.01)
    return LadderThresholds(
        deep_confidence_threshold=round(deep, 4),
        verified_confidence_threshold=round(verified, 4),
    )


def evaluate_ladder_threshold_candidate(
    candidate: LadderThresholds,
    *,
    baseline_score: float = 1.0,
) -> dict[str, float | str]:
    validate_ladder_threshold_overlay(candidate.to_payload())
    return {
        "eval_kind": "deterministic_routing_fixture",
        "score": 1.0,
        "baseline": baseline_score,
    }


def active_ladder_thresholds(settings: AprilSettings) -> dict[str, float]:
    return _active_ladder_thresholds(settings).to_payload()


def validate_ladder_threshold_overlay(payload: dict[str, Any]) -> LadderThresholds:
    allowed = {"deep_confidence_threshold", "verified_confidence_threshold"}
    extra = set(payload) - allowed
    if extra:
        raise ValueError(f"ladder threshold overlay has unsupported keys: {sorted(extra)}")
    missing = allowed - set(payload)
    if missing:
        raise ValueError(f"ladder threshold overlay is missing keys: {sorted(missing)}")
    try:
        deep = float(payload["deep_confidence_threshold"])
        verified = float(payload["verified_confidence_threshold"])
    except (TypeError, ValueError) as exc:
        raise ValueError("ladder threshold overlay values must be floats") from exc
    if not (0.0 < deep < verified < 1.0):
        raise ValueError("ladder threshold overlay must satisfy 0 < deep < verified < 1")
    if not (0.2 <= deep <= 0.6):
        raise ValueError("deep_confidence_threshold overlay must be within [0.2, 0.6]")
    if not (0.5 <= verified <= 0.9):
        raise ValueError("verified_confidence_threshold overlay must be within [0.5, 0.9]")
    return LadderThresholds(
        deep_confidence_threshold=deep,
        verified_confidence_threshold=verified,
    )


def _active_ladder_thresholds(
    settings: AprilSettings,
    *,
    audit: AuditLogger | None = None,
) -> LadderThresholds:
    base = LadderThresholds(
        deep_confidence_threshold=settings.deep_mode.deep_confidence_threshold,
        verified_confidence_threshold=settings.deep_mode.verified_confidence_threshold,
    )
    pointer = _read_ladder_pointer(settings)
    version = _active_ladder_version(pointer)
    if version is None:
        return base
    path = _ladder_overlay_path(settings, version)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return validate_ladder_threshold_overlay(payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        if audit is not None:
            audit.write(
                {
                    "event_type": "ladder_threshold_overlay_blocked_at_load",
                    "actor": "dreamer",
                    "version": version,
                    "reason": str(exc)[:300],
                }
            )
        return base


def _ladder_config_dir(settings: AprilSettings) -> Path:
    return settings.evolution_path / "config"


def _ladder_overlay_path(settings: AprilSettings, version: int) -> Path:
    return _ladder_config_dir(settings) / f"ladder-v{version:03d}.json"


def _ladder_pointer_path(settings: AprilSettings) -> Path:
    return _ladder_config_dir(settings) / "ladder-active.json"


def _read_ladder_pointer(settings: AprilSettings) -> dict[str, Any] | None:
    try:
        payload = json.loads(_ladder_pointer_path(settings).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _active_ladder_version(pointer: dict[str, Any] | None) -> int | None:
    if pointer is None:
        return None
    value = pointer.get("active_version")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_ladder_version(path: Path) -> int | None:
    match = re.fullmatch(r"ladder-v(\d+)\.json", path.name)
    if match is None:
        return None
    return int(match.group(1))
