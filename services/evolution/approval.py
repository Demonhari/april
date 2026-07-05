from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from april_common.audit import AuditLogger
from april_common.settings import AprilSettings
from services.evolution.evaluator import (
    RuntimeEvalClient,
    evaluate_overlay_candidate,
    evaluate_overlay_candidate_real_runtime,
)
from services.evolution.versions import (
    WRITE_CAPABLE_AGENTS,
    OverlayApplyResult,
    PromptOverlayManager,
)
from services.memory.database import Database

_AGENT_NAME_MAX = 64


@dataclass(frozen=True, slots=True)
class PendingOverlay:
    agent: str
    content_hash: str
    chars: int
    preview: str
    basename: str
    blocked_reason: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "content_hash": self.content_hash,
            "chars": self.chars,
            "preview": self.preview,
            "basename": self.basename,
            "blocked_reason": self.blocked_reason,
        }


class PromptOverlayApprovalService:
    """Review/approve path for overlays that returned approval_required.

    The Dreamer persists every generated overlay candidate as plain text under
    data/evolution/candidates/. Write-capable agents (Forge/Hand) never
    auto-apply, so their candidates wait here until the local user explicitly
    approves the exact bytes (matched by SHA-256). Approval re-runs the same
    policy checks as auto-apply: structural tool/permission content is always
    rejected, the eval baseline still applies, and nothing outside prompt
    overlays can ever be changed through this path.
    """

    def __init__(
        self,
        settings: AprilSettings,
        database: Database,
        *,
        audit: AuditLogger | None = None,
        runtime_client: RuntimeEvalClient | None = None,
    ) -> None:
        self.settings = settings
        self.database = database
        self.audit = audit
        self.runtime_client = runtime_client
        self.manager = PromptOverlayManager(settings, database, audit=audit)

    async def list_pending(self) -> list[PendingOverlay]:
        candidates_dir = self.settings.evolution_path / "candidates"
        if not candidates_dir.is_dir():
            return []
        applied_hashes = {
            str(row["content_hash"])
            for row in await self.database.fetchall("SELECT content_hash FROM prompt_versions")
        }
        pending: list[PendingOverlay] = []
        for path in sorted(candidates_dir.glob("*.overlay.txt")):
            agent = _agent_from_basename(path.name)
            if agent is None or agent not in WRITE_CAPABLE_AGENTS:
                # Non-write-capable agents auto-apply or are discarded during
                # the dream; only write-capable overlays wait for approval.
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except OSError:
                continue
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if content_hash in applied_hashes:
                continue
            blocked = self.manager.rejection_reason(content)
            pending.append(
                PendingOverlay(
                    agent=agent,
                    content_hash=content_hash,
                    chars=len(content),
                    preview=content[:400],
                    basename=path.name,
                    blocked_reason=blocked,
                )
            )
        return pending

    async def approve(self, *, agent: str, content_hash: str) -> OverlayApplyResult:
        if len(agent) > _AGENT_NAME_MAX:
            return OverlayApplyResult("discarded", agent[:_AGENT_NAME_MAX], reason="bad agent")
        if agent not in WRITE_CAPABLE_AGENTS:
            return OverlayApplyResult(
                "discarded",
                agent,
                reason="overlay approval endpoint is limited to write-capable agents",
            )
        content = self._find_candidate(agent=agent, content_hash=content_hash)
        if content is None:
            return OverlayApplyResult(
                "discarded", agent, reason="no pending candidate matches that hash"
            )
        evaluation = evaluate_overlay_candidate(
            agent=agent, content=content, settings=self.settings
        )
        active_score = await self.manager.active_eval_score(agent)
        apply_baseline = max(
            evaluation.baseline,
            active_score if active_score is not None else evaluation.baseline,
        )
        if evaluation.score >= apply_baseline and self.settings.environment == "production":
            real_eval = await evaluate_overlay_candidate_real_runtime(
                agent=agent,
                content=content,
                settings=self.settings,
                runtime_client=self.runtime_client,
            )
            if not real_eval.passed:
                reason = (
                    "real-runtime evaluation required before production activation: "
                    + (
                        "; ".join(real_eval.blockers)
                        if real_eval.blockers
                        else real_eval.status
                    )
                )
                self._audit(
                    "prompt_overlay_user_approval_blocked",
                    agent=agent,
                    detail=f"status={real_eval.status} hash={content_hash[:12]}",
                )
                return OverlayApplyResult("pending_real_runtime", agent, reason=reason)
        result = await self.manager.apply_candidate(
            agent=agent,
            content=content,
            eval_score=evaluation.score,
            baseline_score=apply_baseline,
            source="dreamer",
            approved=True,
        )
        self._audit(
            "prompt_overlay_user_approval",
            agent=agent,
            detail=f"status={result.status} hash={content_hash[:12]}",
        )
        return result

    def _find_candidate(self, *, agent: str, content_hash: str) -> str | None:
        candidates_dir = self.settings.evolution_path / "candidates"
        if not candidates_dir.is_dir():
            return None
        for path in sorted(candidates_dir.glob("*.overlay.txt")):
            if _agent_from_basename(path.name) != agent:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except OSError:
                continue
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if digest == content_hash:
                return content
        return None

    def _audit(self, event_type: str, *, agent: str, detail: str) -> None:
        if self.audit is not None:
            self.audit.write(
                {
                    "event_type": event_type,
                    "actor": "local-user",
                    "agent": agent,
                    "detail": detail,
                }
            )


def _agent_from_basename(basename: str) -> str | None:
    # Candidate files are written as f"{agent}-{index}.overlay.txt".
    stem = basename.removesuffix(".overlay.txt")
    agent, separator, index = stem.rpartition("-")
    if not separator or not agent or not index.isdigit():
        return None
    return agent
