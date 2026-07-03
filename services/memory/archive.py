from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from agents.memory import ArchiveAgent, ArchiveMemoryCandidate
from april_common.audit import AuditLogger
from april_common.settings import AprilSettings
from services.april_runtime.client import RuntimeClient
from services.memory.schemas import MemoryRecord, Message, VectorMetadata
from services.memory.sqlite_memory import SqliteMemory
from services.memory.vector_memory import VectorMemory

ARCHIVE_SOURCE = "archive"
MIN_ARCHIVE_CONFIDENCE = 0.65
_NEGATION_TOKENS = {"not", "no", "never", "don't", "dont", "doesn't", "doesnt", "isn't", "isnt"}


@dataclass(frozen=True, slots=True)
class ArchiveWriteOutcome:
    status: Literal["written", "duplicate", "low_confidence", "daily_cap", "contradiction"]
    candidate: ArchiveMemoryCandidate
    memory: MemoryRecord | None = None
    detail: str | None = None


class ArchiveMemoryWriter:
    def __init__(
        self,
        memory: SqliteMemory,
        *,
        vector_memory: VectorMemory | None = None,
        audit: AuditLogger | None = None,
        min_confidence: float = MIN_ARCHIVE_CONFIDENCE,
        daily_cap: int = 30,
    ) -> None:
        self.memory = memory
        self.vector_memory = vector_memory
        self.audit = audit
        self.min_confidence = min_confidence
        self.daily_cap = daily_cap

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
        written_today = await self.memory.count_machine_memories_since(
            _today_start_iso(), source=ARCHIVE_SOURCE
        )
        if written_today >= self.daily_cap:
            return self._discard("daily_cap", candidate, "daily archive memory cap reached")
        duplicate = await self.memory.find_duplicate_memory(
            candidate.content,
            kind=candidate.kind,
            project_id=project_id,
        )
        if duplicate is not None:
            self._audit(
                "archive_memory_duplicate",
                candidate,
                source_session_id=source_session_id,
                memory_id=duplicate.id,
            )
            return ArchiveWriteOutcome("duplicate", candidate, memory=duplicate)
        contradiction = await self._contradiction(candidate, project_id=project_id)
        if contradiction is not None:
            return self._discard(
                "contradiction",
                candidate,
                f"possible contradiction with memory {contradiction.id}",
            )
        reason = f"{candidate.reason} (source_session={source_session_id})"
        record = await self.memory.create_memory(
            candidate.content,
            kind=candidate.kind,
            reason=reason,
            project_id=project_id,
            confidence=candidate.confidence,
            source=ARCHIVE_SOURCE,
        )
        self._index_memory(record)
        self._audit(
            "archive_memory_written",
            candidate,
            source_session_id=source_session_id,
            memory_id=record.id,
        )
        return ArchiveWriteOutcome("written", candidate, memory=record)

    async def _contradiction(
        self, candidate: ArchiveMemoryCandidate, *, project_id: str | None
    ) -> MemoryRecord | None:
        candidate_norm = _normalized(candidate.content)
        candidate_negated = _has_negation(candidate_norm)
        for existing in await self.memory.list_memories(project_id=project_id):
            if existing.kind != candidate.kind:
                continue
            existing_norm = _normalized(existing.content)
            if _without_negation(existing_norm) != _without_negation(candidate_norm):
                continue
            if _has_negation(existing_norm) != candidate_negated:
                return existing
        return None

    def _discard(
        self,
        status: Literal["low_confidence", "daily_cap", "contradiction"],
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
        writer: ArchiveMemoryWriter | None = None,
    ) -> None:
        self.settings = settings
        self.memory = memory
        self.archive_agent = archive_agent or ArchiveAgent(
            runtime_client,
            model_id=settings.brain.model_id,
        )
        self.writer = writer or ArchiveMemoryWriter(
            memory,
            vector_memory=vector_memory,
            audit=audit,
            daily_cap=settings.evolution.daily_memory_cap,
        )

    async def reflect_session(self, session_id: str) -> list[ArchiveWriteOutcome]:
        session = await self.memory.get_session(session_id)
        if session is None or session.conversation_id is None:
            return []
        messages = await self.memory.recent_messages(session.conversation_id, limit=50)
        transcript = _session_transcript(messages)
        if not transcript.strip():
            return []
        candidates = await self.archive_agent.extract(
            transcript,
            request_id=f"archive-{session_id}",
        )
        return await self.writer.write_candidates(
            candidates,
            source_session_id=session_id,
            project_id=None,
        )


def _session_transcript(messages: list[Message]) -> str:
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


def _today_start_iso() -> str:
    now = datetime.now(UTC)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _normalized(value: str) -> str:
    return " ".join(value.casefold().replace(".", " ").replace(",", " ").split())


def _has_negation(value: str) -> bool:
    return any(token in _NEGATION_TOKENS for token in value.split())


def _without_negation(value: str) -> str:
    return " ".join(token for token in value.split() if token not in _NEGATION_TOKENS)
