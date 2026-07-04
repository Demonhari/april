from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
    status: Literal["applied", "discarded", "approval_required"]
    agent: str
    version: int | None = None
    reason: str | None = None
    path: Path | None = None


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
