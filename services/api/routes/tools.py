from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from fastapi import Depends, FastAPI, Header

from april_common.errors import PermissionDeniedError
from services.api.dependencies import ApiContainer
from services.api.schemas import (
    AgentRunRequest,
    ChatResponse,
    ToolApprovalAction,
    ToolRequestEnvelope,
)
from skills.playbooks import PlaybookRunner


def register_tool_routes(app: FastAPI, authorized: Callable[..., Any]) -> None:
    @app.post("/agents/run")
    async def agents_run(
        request: AgentRunRequest,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> ChatResponse:
        request_id = x_request_id or str(uuid.uuid4())
        if not request.options.structured:
            raise PermissionDeniedError(
                "Direct agent runs only support structured execution.",
                {"agent": request.agent},
            )
        async with active.require_session_manager().interaction(request.conversation_id):
            result = await active.orchestrator.run_agent(
                agent_id=request.agent,
                message=request.message,
                conversation_id=request.conversation_id,
                request_id=request_id,
                project_id=request.project_id,
                repo_path=request.repo_path,
            )
        return ChatResponse(request_id=request_id, result=result)

    @app.post("/tools/request")
    async def tool_request(
        request: ToolRequestEnvelope,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        context = await active.tool_executor.context(
            request_id=x_request_id or str(uuid.uuid4()),
            actor="local-user",
            agent_id=request.agent,
            project_id=str(request.args["project_id"]) if request.args.get("project_id") else None,
            source="api",
        )
        outcome = await active.tool_executor.request_or_execute(
            tool=request.tool,
            args=request.args,
            context=context,
        )
        if outcome.approval is not None:
            return {"status": "pending_approval", "approval": outcome.approval}
        return {"status": outcome.status, "result": outcome.result}

    @app.post("/tools/approve")
    async def approve(
        request: ToolApprovalAction,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        request_id = x_request_id or str(uuid.uuid4())
        approval = await active.approvals.get(request.approval_id)
        playbook_run_id = approval.metadata.get("playbook_run_id")
        if isinstance(playbook_run_id, str):
            if request.tool is not None:
                raise PermissionDeniedError(
                    "Playbook approvals resume only their persisted exact action."
                )
            result = await PlaybookRunner(active.tool_executor, memory=active.memory).resume(
                playbook_run_id,
                approval_id=request.approval_id,
                actor="local-user",
            )
            return {"playbook_run": asdict(result)}
        return await active.orchestrator.approve_tool(
            approval_id=request.approval_id,
            actor="local-user",
            request_id=request_id,
            tool=request.tool,
            args=request.args if request.tool is not None else None,
        )

    @app.post("/tools/deny")
    async def deny(
        request: ToolApprovalAction,
        active: ApiContainer = Depends(authorized),
        x_request_id: str | None = Header(default=None),
    ) -> object:
        approval = await active.approvals.get(request.approval_id)
        result = await active.orchestrator.deny_tool(
            approval_id=request.approval_id,
            actor="local-user",
            request_id=x_request_id or str(uuid.uuid4()),
        )
        playbook_run_id = approval.metadata.get("playbook_run_id")
        if isinstance(playbook_run_id, str):
            playbook = await PlaybookRunner(
                active.tool_executor,
                memory=active.memory,
            ).mark_denied(playbook_run_id, approval_id=request.approval_id)
            return {"approval": result, "playbook_run": asdict(playbook)}
        return result

    @app.get("/approvals")
    async def approvals(active: ApiContainer = Depends(authorized)) -> object:
        return {
            "approvals": [record.model_dump() for record in await active.approvals.list_pending()]
        }
