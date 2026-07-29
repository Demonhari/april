from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI

from april_common.errors import PermissionDeniedError
from april_common.path_security import PathPolicy, normalize_existing_path
from april_common.settings import AprilSettings
from services.api.dependencies import ApiContainer
from services.api.schemas import DocumentCreateRequest, ProjectCreateRequest


def register_project_routes(
    app: FastAPI,
    authorized: Callable[..., Any],
) -> None:
    @app.delete("/conversations/{conversation_id}")
    async def conversation_delete(
        conversation_id: str, active: ApiContainer = Depends(authorized)
    ) -> object:
        return {"deleted": await active.memory.delete_conversation(conversation_id)}

    @app.get("/projects")
    async def projects(active: ApiContainer = Depends(authorized)) -> object:
        return {
            "projects": [project.model_dump() for project in await active.memory.list_projects()]
        }

    @app.post("/projects")
    async def project_add(
        request: ProjectCreateRequest, active: ApiContainer = Depends(authorized)
    ) -> object:
        normalized = _normalize_project_path(request.path, active.settings)
        project = await active.memory.add_project(str(normalized), name=request.name)
        return project

    @app.post("/projects/{project_id}/index", status_code=202)
    async def project_index(project_id: str, active: ApiContainer = Depends(authorized)) -> object:
        project = await active.memory.get_project(project_id)
        if project is None:
            raise PermissionDeniedError("Project not found.")
        if active.job_store is None:
            raise PermissionDeniedError("Durable job store is unavailable.")
        job = await active.job_store.submit(
            job_type="repository_index",
            payload={"repo_path": project.path, "project_id": project_id},
            owner="local-user",
            project_id=project_id,
        )
        return job.model_dump(mode="json")

    @app.post("/documents", status_code=202)
    async def document_add(
        request: DocumentCreateRequest, active: ApiContainer = Depends(authorized)
    ) -> object:
        folder = _normalize_project_path(request.path, active.settings)
        if active.job_store is None:
            raise PermissionDeniedError("Durable job store is unavailable.")
        job = await active.job_store.submit(
            job_type="document_index",
            payload={"folder_path": str(folder)},
            owner="local-user",
        )
        return job.model_dump(mode="json")

    @app.get("/documents")
    async def documents(active: ApiContainer = Depends(authorized)) -> object:
        return {"documents": active.vector_memory.sources(source_type="document")}

    @app.get("/documents/search")
    async def documents_search(q: str, active: ApiContainer = Depends(authorized)) -> object:
        chunks = active.memory_retriever.document_chunks(q)
        return {
            "chunks": [chunk.model_dump() for chunk in chunks],
            "citations": [
                {
                    "path": chunk.metadata.get("path"),
                    "start_line": chunk.metadata.get("start_line"),
                    "end_line": chunk.metadata.get("end_line"),
                }
                for chunk in chunks
                if chunk.metadata.get("path")
            ],
        }


def _normalize_project_path(path: str, settings: AprilSettings) -> Path:
    policy = PathPolicy(
        allowed_roots=tuple(settings.allowed_roots),
        max_read_bytes=settings.paths.max_file_read_bytes,
        max_write_bytes=settings.paths.max_file_write_bytes,
    )
    normalized = normalize_existing_path(path, policy)
    if not normalized.is_dir():
        raise PermissionDeniedError("Project path must be an existing directory.")
    return normalized
