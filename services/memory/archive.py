from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from agents.memory import ArchiveAgent, ArchiveMemoryCandidate
from april_common.audit import AuditLogger
from april_common.settings import AprilSettings
from services.april_runtime.client import RuntimeClient
from services.memory.policy import MemoryPolicy
from services.memory.repository import MemoryRepository
from services.memory.schemas import MemoryRecord, Message, VectorMetadata
from services.memory.sqlite_memory import SqliteMemory
from services.memory.vector_memory import VectorMemory

# v2 source vocabulary: session reflection writes rows as "reflection".
# "archive" is the legacy spelling; existing rows keep it and the daily cap
# counts both so old databases never exceed the budget after an upgrade.
ARCHIVE_SOURCE = "reflection"
LEGACY_ARCHIVE_SOURCE = "archive"
# Candidates below this confidence are discarded (architecture: 0.5); override
# via ArchiveMemoryWriter(min_confidence=...) / evolution.archive_min_confidence.
MIN_ARCHIVE_CONFIDENCE = 0.5
# Cosine similarity above which a vector hit of the same kind is treated as a
# near-duplicate and merged instead of creating a new row.
NEAR_DUPLICATE_SIMILARITY = 0.92
_NEGATION_TOKENS = {"not", "no", "never", "don't", "dont", "doesn't", "doesnt", "isn't", "isnt"}
# Copulas/verbs that split a statement into subject and value for the
# deterministic value-mismatch contradiction check ("editor is vim" vs
# "editor is emacs"). Only the first occurrence splits.
_VALUE_SPLIT_TOKENS = ("is", "are", "prefers", "prefer", "uses", "use", "likes", "wants")


@dataclass(frozen=True, slots=True)
class ArchiveWriteOutcome:
    status: Literal[
        "written", "duplicate", "low_confidence", "daily_cap", "contradiction", "sensitive"
    ]
    candidate: ArchiveMemoryCandidate
    memory: MemoryRecord | None = None
    detail: str | None = None


class ArchiveMemoryWriter:
    def __init__(
        self,
        memory: SqliteMemory,
        *,
        vector_memory: VectorMemory | None = None,
        repository: MemoryRepository | None = None,
        audit: AuditLogger | None = None,
        min_confidence: float = MIN_ARCHIVE_CONFIDENCE,
        daily_cap: int = 30,
        near_duplicate_similarity: float = NEAR_DUPLICATE_SIMILARITY,
    ) -> None:
        self.memory = memory
        self.vector_memory = vector_memory
        self.repository = repository
        self.audit = audit
        self.min_confidence = min_confidence
        self.daily_cap = daily_cap
        self.near_duplicate_similarity = near_duplicate_similarity
        self.policy = MemoryPolicy()

    async def write_candidates(
        self,
        candidates: list[ArchiveMemoryCandidate],
        *,
        source_session_id: str,
        project_id: str | None = None,
    ) -> list[ArchiveWriteOutcome]:
        outcomes: list[ArchiveWriteOutcome] = []
        for candidate in candidates:
            outcome = await self._write_one(
                candidate,
                source_session_id=source_session_id,
                project_id=project_id,
            )
            outcomes.append(outcome)
        return outcomes

    async def _write_one(
        self,
        candidate: ArchiveMemoryCandidate,
        *,
        source_session_id: str,
        project_id: str | None,
    ) -> ArchiveWriteOutcome:
        if candidate.confidence < self.min_confidence:
            return self._discard("low_confidence", candidate, "confidence below threshold")
        if self.policy.is_sensitive(candidate.content):
            return self._discard(
                "sensitive", candidate, "sensitive-looking content is never stored"
            )
        written_today = await self.memory.count_machine_memories_since(
            _today_start_iso(), source=(ARCHIVE_SOURCE, LEGACY_ARCHIVE_SOURCE)
        )
        if written_today >= self.daily_cap:
            return self._discard("daily_cap", candidate, "daily archive memory cap reached")
        duplicate = await self._find_duplicate(candidate, project_id=project_id)
        if duplicate is not None:
            existing, how = duplicate
            # Duplicates merge instead of piling up: the existing row's usage is
            # refreshed and its confidence keeps the maximum of both statements.
            refreshed = await (
                self.repository.refresh_memory(existing.id, confidence=candidate.confidence)
                if self.repository is not None
                else self.memory.refresh_memory(existing.id, confidence=candidate.confidence)
            )
            self._audit(
                "archive_memory_duplicate",
                candidate,
                source_session_id=source_session_id,
                memory_id=existing.id,
                detail=how,
            )
            return ArchiveWriteOutcome(
                "duplicate", candidate, memory=refreshed or existing, detail=how
            )
        contradiction = await self._contradiction(candidate, project_id=project_id)
        reason = f"{candidate.reason} (source_session={source_session_id})"
        create = (
            self.repository.create_memory
            if self.repository is not None
            else self.memory.create_memory
        )
        record = await create(
            candidate.content,
            kind=candidate.kind,
            reason=reason,
            project_id=project_id,
            confidence=candidate.confidence,
            source=ARCHIVE_SOURCE,
        )
        if self.repository is None:
            self._index_memory(record)
        else:
            session = await self.memory.get_session(source_session_id)
            messages = (
                await self.memory.recent_messages(session.conversation_id, limit=200)
                if session is not None and session.conversation_id is not None
                else []
            )
            await self.repository.set_provenance(
                record.id,
                source_session_id=source_session_id,
                source_conversation_id=(session.conversation_id if session is not None else None),
                source_message_ids=[message.id for message in messages],
            )
        if contradiction is not None:
            # Keep both statements and flag the pair; the Dreamer adjudicates
            # later instead of the writer silently discarding the candidate.
            pair = await self.memory.record_memory_contradiction(
                memory_id_a=contradiction.id,
                memory_id_b=record.id,
            )
            detail = (
                f"kept both; contradiction pair {pair.id} with memory "
                f"{contradiction.id} flagged for Dreamer adjudication"
            )
            self._audit(
                "archive_memory_contradiction",
                candidate,
                source_session_id=source_session_id,
                memory_id=record.id,
                detail=detail,
            )
            return ArchiveWriteOutcome("contradiction", candidate, memory=record, detail=detail)
        self._audit(
            "archive_memory_written",
            candidate,
            source_session_id=source_session_id,
            memory_id=record.id,
        )
        return ArchiveWriteOutcome("written", candidate, memory=record)

    async def _find_duplicate(
        self, candidate: ArchiveMemoryCandidate, *, project_id: str | None
    ) -> tuple[MemoryRecord, str] | None:
        """Exact-normalized duplicate first, then vector near-duplicate if available."""
        exact = await self.memory.find_duplicate_memory(
            candidate.content,
            kind=candidate.kind,
            project_id=project_id,
        )
        if exact is not None:
            return exact, "exact_duplicate_merged"
        if self.vector_memory is None:
            return None
        try:
            hits = self.vector_memory.search(candidate.content, limit=3, source_type="memory")
        except Exception:
            return None
        for hit in hits:
            if hit.score < self.near_duplicate_similarity:
                continue
            record = await self.memory.get_memory(hit.id)
            if record is None or record.kind != candidate.kind:
                continue
            if record.project_id != project_id:
                continue
            # A near-duplicate that flips negation is a contradiction, not a merge.
            if _has_negation(_normalized(record.content)) != _has_negation(
                _normalized(candidate.content)
            ):
                continue
            return record, "near_duplicate_merged"
        return None

    async def _contradiction(
        self, candidate: ArchiveMemoryCandidate, *, project_id: str | None
    ) -> MemoryRecord | None:
        """Deterministic contradiction detection: negation flips and value mismatches.

        Detection only ever flags a pair for Dreamer adjudication; it never
        deletes or supersedes existing memory on its own.
        """
        candidate_norm = _normalized(candidate.content)
        candidate_negated = _has_negation(candidate_norm)
        for existing in await self.memory.list_memories(project_id=project_id):
            if existing.kind != candidate.kind:
                continue
            existing_norm = _normalized(existing.content)
            if _without_negation(existing_norm) == _without_negation(candidate_norm):
                if _has_negation(existing_norm) != candidate_negated:
                    return existing
                continue
            if _value_mismatch(existing_norm, candidate_norm):
                return existing
        return None

    def _discard(
        self,
        status: Literal["low_confidence", "daily_cap", "sensitive"],
        candidate: ArchiveMemoryCandidate,
        detail: str,
    ) -> ArchiveWriteOutcome:
        self._audit(
            f"archive_memory_{status}",
            candidate,
            source_session_id=None,
            detail=detail,
        )
        return ArchiveWriteOutcome(status, candidate, detail=detail)

    def _index_memory(self, record: MemoryRecord) -> None:
        if self.vector_memory is None:
            return
        content_hash = hashlib.sha256(record.content.encode("utf-8")).hexdigest()
        self.vector_memory.upsert(
            record_id=record.id,
            content=record.content,
            metadata=VectorMetadata(
                source_type="memory",
                source_id=record.id,
                project_id=record.project_id,
                content_hash=content_hash,
                created_at=record.created_at,
            ),
        )

    def _audit(
        self,
        event_type: str,
        candidate: ArchiveMemoryCandidate,
        *,
        source_session_id: str | None,
        memory_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        if self.audit is None:
            return
        self.audit.write(
            {
                "event_type": event_type,
                "actor": "archive_agent",
                "source_session_id": source_session_id,
                "memory_id": memory_id,
                "kind": candidate.kind,
                "confidence": candidate.confidence,
                "content_length": len(candidate.content),
                "detail": detail,
            }
        )


class ArchiveReflectionService:
    def __init__(
        self,
        settings: AprilSettings,
        *,
        memory: SqliteMemory,
        runtime_client: RuntimeClient,
        vector_memory: VectorMemory | None = None,
        audit: AuditLogger | None = None,
        archive_agent: ArchiveAgent | None = None,
        archive_model_id: str | None = None,
        writer: ArchiveMemoryWriter | None = None,
        repository: MemoryRepository | None = None,
    ) -> None:
        self.settings = settings
        self.memory = memory
        self.archive_agent = archive_agent or ArchiveAgent(
            runtime_client,
            model_id=archive_model_id or settings.brain.model_id,
        )
        self.writer = writer or ArchiveMemoryWriter(
            memory,
            vector_memory=vector_memory,
            repository=repository,
            audit=audit,
            min_confidence=settings.evolution.archive_min_confidence,
            daily_cap=settings.evolution.daily_memory_cap,
        )

    async def reflect_session(self, session_id: str) -> list[ArchiveWriteOutcome]:
        session = await self.memory.get_session(session_id)
        if session is None or session.conversation_id is None:
            return []
        conversation = await self.memory.get_conversation(session.conversation_id)
        project_id = conversation.project_id if conversation is not None else None
        messages = await self.memory.recent_messages(session.conversation_id, limit=50)
        transcript = _session_transcript(messages)
        if not transcript.strip():
            return []
        # Local context beyond raw messages helps the Archive judge what stuck:
        # sanitized tool-call summaries (never args or outputs, so no secrets)
        # and feedback/approval outcomes tied to this conversation.
        tool_calls = await self.memory.list_tool_call_summaries(
            conversation_id=session.conversation_id
        )
        feedback = await self.memory.list_feedback_events(conversation_id=session.conversation_id)
        sections = [transcript]
        if tool_calls:
            sections.append(
                "Tool calls this session (name, status, risk only):\n"
                + "\n".join(
                    f"- {call['tool']}: {call['status']} ({call['risk_level']})"
                    for call in tool_calls
                )
            )
        if feedback:
            sections.append(
                "User feedback this session:\n"
                + "\n".join(
                    f"- {event.rating}" + (f": {event.reason}" if event.reason else "")
                    for event in feedback
                )
            )
        candidates = await self.archive_agent.extract(
            "\n\n".join(sections),
            request_id=f"archive-{session_id}",
        )
        return await self.writer.write_candidates(
            candidates,
            source_session_id=session_id,
            project_id=project_id,
        )


def _session_transcript(messages: list[Message]) -> str:
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


def _today_start_iso() -> str:
    now = datetime.now(UTC)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat().replace("+00:00", "Z")


def _normalized(value: str) -> str:
    return " ".join(value.casefold().replace(".", " ").replace(",", " ").split())


def _has_negation(value: str) -> bool:
    return any(token in _NEGATION_TOKENS for token in value.split())


def _without_negation(value: str) -> str:
    return " ".join(token for token in value.split() if token not in _NEGATION_TOKENS)


def _split_subject_value(value: str) -> tuple[str, str] | None:
    """Split "the user's editor is vim" into ("the user's editor", "vim")."""
    tokens = value.split()
    for index, token in enumerate(tokens):
        if token in _VALUE_SPLIT_TOKENS and 0 < index < len(tokens) - 1:
            return " ".join(tokens[:index]), " ".join(tokens[index + 1 :])
    return None


_BARE_SUBJECTS = {"i", "you", "we", "they", "it", "user", "the user"}


def _value_mismatch(existing_norm: str, candidate_norm: str) -> bool:
    """Same specific subject, different stated value ⇒ likely contradiction.

    Both statements must be non-negated (negation pairs are handled separately)
    and split cleanly around a copula/preference verb with identical subjects.
    A bare-pronoun subject ("i prefer …") is never enough: two preferences can
    coexist, and a false contradiction pair would let the Dreamer supersede a
    valid memory during adjudication.
    """
    if _has_negation(existing_norm) or _has_negation(candidate_norm):
        return False
    existing_split = _split_subject_value(existing_norm)
    candidate_split = _split_subject_value(candidate_norm)
    if existing_split is None or candidate_split is None:
        return False
    existing_subject, existing_value = existing_split
    candidate_subject, candidate_value = candidate_split
    if existing_subject != candidate_subject or existing_value == candidate_value:
        return False
    subject_tokens = existing_subject.split()
    meaningful = [token for token in subject_tokens if token not in {"the", "a", "an", "my"}]
    if existing_subject in _BARE_SUBJECTS or len(meaningful) < 1:
        return False
    # Require at least one non-pronoun content token in the subject.
    return any(token not in _BARE_SUBJECTS and len(token) >= 3 for token in meaningful)
