from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException

from april_common.errors import PermissionDeniedError
from services.api.dependencies import ApiContainer
from services.api.schemas import MemoryCreateRequest
from services.memory.writer import MemoryWriter


def register_memory_routes(app: FastAPI, authorized: Callable[..., Any]) -> None:
    @app.post("/memory")
    async def memory_create(
        request: MemoryCreateRequest,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
    ) -> object:
        if (
            request.project_id is not None
            and await active.memory.get_project(request.project_id) is None
        ):
            raise PermissionDeniedError(
                "Unknown project for project-scoped memory.",
                {"project_id": request.project_id},
            )
        if request.source_conversation_id is not None:
            conversation = await active.memory.get_conversation(request.source_conversation_id)
            if conversation is None:
                raise PermissionDeniedError(
                    "Unknown source conversation for memory write.",
                    {"conversation_id": request.source_conversation_id},
                )
            if conversation.project_id != request.project_id:
                raise PermissionDeniedError(
                    "Memory source conversation project scope does not match.",
                    {
                        "conversation_project_id": conversation.project_id,
                        "memory_project_id": request.project_id,
                    },
                )
        writer = MemoryWriter(
            active.memory_repository,
            sensitive_encryption_available=(
                active.settings.memory.sensitive_encryption_enabled
                and active.memory.sensitive_encryption is not None
            ),
        )
        record = await writer.write(
            request.content,
            reason=request.reason,
            memory_type=request.memory_type,
            requested_by_user=True,
            project_id=request.project_id,
            sensitive=request.sensitive,
        )
        await active.memory_repository.set_provenance(
            record.id,
            source_conversation_id=request.source_conversation_id,
        )
        index_health = await active.memory_repository.health()
        active.approvals.audit.write(
            {
                "event_type": "memory_written",
                "request_id": x_request_id or str(uuid.uuid4()),
                "actor": "local-user",
                "memory_id": record.id,
                "memory_type": record.kind,
                "project_id": record.project_id,
                "source_conversation_id": request.source_conversation_id,
                "content_length": len(record.content),
                "reason_length": len(record.reason),
            }
        )
        return {
            "memory": record.model_dump(),
            "stored": f"Stored {record.kind} memory.",
            "index_repair_required": index_health.repair_required,
        }

    @app.get("/memory/search")
    async def memory_search(
        q: str,
        project_id: str | None = None,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        if project_id is not None and await active.memory.get_project(project_id) is None:
            raise PermissionDeniedError(
                "Unknown project for memory search.",
                {"project_id": project_id},
            )
        results = await active.memory.search_memories(q, project_id=project_id)
        return {"results": [result.model_dump() for result in results]}

    @app.delete("/memory/{memory_id}")
    async def memory_delete(
        memory_id: str,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        deleted = await active.memory_repository.delete_memory(memory_id)
        index_health = await active.memory_repository.health()
        return {
            "deleted": deleted,
            "index_repair_required": index_health.repair_required,
        }

    @app.get("/memory/inspect")
    async def memory_inspect(
        state: str = "machine",
        limit: int = 100,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        try:
            records = await active.memory.list_memories_by_state(state, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "state": state,
            "memories": [record.model_dump() for record in records],
        }

    @app.get("/memory/export")
    async def memory_export(
        project_id: str | None = None,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        if project_id is not None and await active.memory.get_project(project_id) is None:
            raise PermissionDeniedError(
                "Unknown project for memory export.",
                {"project_id": project_id},
            )
        return {"export": await active.memory.export_memories(project_id=project_id)}

    @app.post("/memory/reindex")
    async def memory_reindex(active: ApiContainer = Depends(authorized)) -> object:
        reindexed = await active.memory_repository.rebuild()
        configured_provider = active.settings.memory.embedding_provider
        active_provider = active.vector_memory.embedding.name
        vector_health = active.vector_memory.health()
        fallback_active = configured_provider == "runtime-local" and (
            active_provider == "hashed-token"
        )
        compatible = bool(vector_health.get("compatible", True))
        return {
            "reindexed": reindexed,
            "provider": active_provider,
            "configured_provider": configured_provider,
            "dimensions": active.vector_memory.embedding.dimensions,
            "index_compatible": compatible,
            "fallback_active": fallback_active,
            "degraded": fallback_active or not compatible,
        }

    @app.post("/memory/repair-index")
    async def memory_repair_index(
        apply: bool = False,
        active: ApiContainer = Depends(authorized),
    ) -> object:
        return active.vector_memory.repair_index(apply=apply)
