"""Stage negative feedback signals as pending evolution eval cases.

Explicit "bad" feedback, implicit correction turns, and approval denials are
already recorded as feedback events. This module turns those signals into
reviewable eval-case YAML files under the fenced evolution data path
(``data/evolution/evals/pending/``) so the Dreamer can report them and a human
can promote them into real eval fixtures.

Invariants:

* Staging never mutates repository tests — cases land only under the
  :class:`EvolutionWriteGuard`-fenced evolution data path.
* Sensitive-looking content is excluded per the existing memory policy; a case
  that cannot be captured safely is skipped, never stored redacted-in-name-only.
* Staging is best-effort: callers swallow failures so feedback capture can
  never break the chat/denial flow itself.
"""

from __future__ import annotations

import logging
from pathlib import Path

from april_common.audit import AuditLogger
from april_common.settings import AprilSettings
from services.evolution.evaluator import write_pending_eval_case
from services.evolution.write_guard import EvolutionWriteGuard
from services.memory.policy import MemoryPolicy
from services.memory.schemas import FeedbackEventRecord
from services.memory.sqlite_memory import SqliteMemory

logger = logging.getLogger(__name__)

# Bounded excerpts keep staged cases reviewable and avoid archiving whole
# transcripts into the evolution data path.
_MAX_EXCERPT_CHARS = 2_000


async def stage_feedback_eval_case(
    settings: AprilSettings,
    memory: SqliteMemory,
    record: FeedbackEventRecord,
    *,
    kind: str,
    audit: AuditLogger | None = None,
    guard: EvolutionWriteGuard | None = None,
) -> Path | None:
    """Stage one pending eval case from a negative feedback event.

    Returns the staged path, or ``None`` when there is not enough safe context
    (no conversation, no user/assistant exchange, or sensitive content).
    """
    if record.rating != "bad" or record.conversation_id is None:
        return None
    messages = await memory.recent_messages(record.conversation_id, limit=10)
    assistant_index: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "assistant":
            assistant_index = index
            break
    if assistant_index is not None:
        assistant_message: str | None = messages[assistant_index].content
        preceding = messages[:assistant_index]
    else:
        # An approval denial can happen before any assistant reply exists (the
        # run suspended on the approval); the proposal itself is the signal.
        assistant_message = None
        preceding = messages
    user_message = next(
        (message.content for message in reversed(preceding) if message.role == "user"),
        None,
    )
    if user_message is None:
        return None
    if assistant_message is None and kind != "approval_denied":
        return None
    policy = MemoryPolicy()
    reason = record.reason or ""
    if any(policy.is_sensitive(text) for text in (user_message, assistant_message or "", reason)):
        # Sensitive content is excluded entirely per the existing memory policy.
        if audit is not None:
            audit.write(
                {
                    "event_type": "feedback_eval_case_skipped",
                    "actor": "feedback_eval",
                    "kind": kind,
                    "detail": "sensitive content excluded by policy",
                    "conversation_id": record.conversation_id,
                }
            )
        return None
    case = {
        "case_type": "negative_feedback",
        "signal": kind,
        "status": "pending_review",
        "created_at": record.created_at,
        "conversation_id": record.conversation_id,
        "agent_run_id": record.agent_run_id,
        "reason": reason[:_MAX_EXCERPT_CHARS],
        "prompt": user_message[:_MAX_EXCERPT_CHARS],
        "bad_response_excerpt": (
            assistant_message[:_MAX_EXCERPT_CHARS] if assistant_message is not None else None
        ),
        # Human review decides the expected behaviour; the machine never
        # invents an expectation from a single negative signal.
        "expected_behavior": None,
    }
    path = write_pending_eval_case(settings, case, guard=guard)
    if audit is not None:
        audit.write(
            {
                "event_type": "feedback_eval_case_staged",
                "actor": "feedback_eval",
                "kind": kind,
                "path_basename": path.name,
                "conversation_id": record.conversation_id,
                "agent_run_bound": record.agent_run_id is not None,
            }
        )
    return path


def count_pending_eval_cases(settings: AprilSettings) -> int:
    """Count staged, not-yet-reviewed eval cases (for Dreamer reports/doctor)."""
    pending_dir = settings.evolution_path / "evals" / "pending"
    if not pending_dir.is_dir():
        return 0
    return sum(1 for path in pending_dir.glob("*.yaml") if path.is_file())


def list_pending_eval_cases(settings: AprilSettings) -> list[Path]:
    pending_dir = settings.evolution_path / "evals" / "pending"
    if not pending_dir.is_dir():
        return []
    return sorted(path for path in pending_dir.glob("*.yaml") if path.is_file())
