from __future__ import annotations

import asyncio

import pytest

from agents.base import USER_FACING_ASSISTANT_NAME, USER_FACING_IDENTITY_RULE
from agents.registry import default_agent_registry
from april_common.settings import project_root
from services.april_runtime.model_registry import ModelRegistry
from services.brain.capabilities import (
    collect_runtime_self_evidence,
    is_self_introspection_request,
    trusted_capability_summary,
)
from services.pool.agent_pool import CALL_SIGNS
from skills.registry import default_registry
from tests.conftest import FakeRuntimeClient


def test_interactive_prompts_use_april_and_keep_internal_call_signs() -> None:
    agents = default_agent_registry()
    for agent_name, call_sign in CALL_SIGNS.items():
        agent = agents.get(agent_name)
        assert agent is not None
        if agent_name != "memory_agent":
            assert call_sign not in agent.system_prompt
            assert "User-facing identity: APRIL" in agent.system_prompt
            normalized_prompt = " ".join(agent.system_prompt.lower().split())
            assert "internal agent names and call signs are implementation metadata only" in (
                normalized_prompt
            )
    assert CALL_SIGNS["general_agent"] == "Prime"


def test_trusted_self_context_uses_registry_roles_and_separates_storage(settings_tmp) -> None:
    registry = ModelRegistry.from_file(
        project_root() / "configs" / "models.yaml", root=project_root()
    )
    summary = trusted_capability_summary(
        settings=settings_tmp,
        agent_registry=default_agent_registry(),
        tool_registry=default_registry(),
        model_registry=registry,
    )
    assert USER_FACING_ASSISTANT_NAME == "APRIL"
    assert USER_FACING_IDENTITY_RULE in summary
    assert "- conversation/brain: april-brain (configured;" in summary
    assert "- coding: april-coding (configured;" in summary
    assert "- reading: april-reading (configured;" in summary
    assert "SQLite-backed durable memory is local storage, not an AI model." in summary
    assert "loaded=unknown; healthy=unknown" in summary
    assert "april-brain (configured; loaded=yes" not in summary


@pytest.mark.asyncio
async def test_runtime_self_context_reports_only_evidenced_state() -> None:
    runtime = FakeRuntimeClient()
    evidence = await collect_runtime_self_evidence(runtime)
    assert evidence == {"models": [], "health_status": "ok"}
    assert is_self_introspection_request("What models are you using?") is True
    assert is_self_introspection_request("Which model is loaded?") is True
    assert is_self_introspection_request("Are your models loaded?") is True
    assert is_self_introspection_request("Which models are configured?") is True
    assert is_self_introspection_request("What can you do?") is False
    assert is_self_introspection_request("Which model does the document use?") is False
    assert is_self_introspection_request("What model is loaded in this quoted text?") is False
    assert is_self_introspection_request("Which model is loaded, and can you plan my day?") is False


@pytest.mark.asyncio
async def test_runtime_self_evidence_cleans_up_failed_sibling() -> None:
    cancelled = asyncio.Event()

    class FailingRuntime:
        async def models(self) -> dict[str, object]:
            raise RuntimeError("offline")

        async def health(self, *, timeout: float | None = None) -> dict[str, str]:
            del timeout
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return {"status": "ok"}

    assert await collect_runtime_self_evidence(FailingRuntime(), timeout_seconds=0.2) is None
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_runtime_self_evidence_cleans_up_timeout_and_caller_cancellation() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class HangingRuntime:
        async def models(self) -> dict[str, object]:
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return {"models": []}

        async def health(self, *, timeout: float | None = None) -> dict[str, str]:
            del timeout
            await asyncio.Future()
            return {"status": "ok"}

    assert await collect_runtime_self_evidence(HangingRuntime(), timeout_seconds=0.01) is None
    assert cancelled.is_set()

    started.clear()
    cancelled.clear()
    pending = asyncio.create_task(
        collect_runtime_self_evidence(HangingRuntime(), timeout_seconds=1)
    )
    await started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_runtime_self_context_distinguishes_loaded_and_health_evidence(settings_tmp) -> None:
    class SnapshotRuntime:
        async def models(self) -> dict[str, object]:
            return {
                "models": [
                    {"id": "april-brain", "state": "loaded", "load_error": None},
                    {"id": "april-coding", "state": "unloaded", "load_error": None},
                    {"id": "april-reading", "state": "error", "load_error": "missing model"},
                ]
            }

        async def health(self, *, timeout: float | None = None) -> dict[str, object]:
            del timeout
            return {"status": "ok"}

    evidence = await collect_runtime_self_evidence(SnapshotRuntime())
    assert evidence is not None
    registry = ModelRegistry.from_file(
        project_root() / "configs" / "models.yaml", root=project_root()
    )
    summary = trusted_capability_summary(
        settings=settings_tmp,
        agent_registry=default_agent_registry(),
        tool_registry=default_registry(),
        model_registry=registry,
        runtime_evidence=evidence,
    )
    assert "conversation/brain: april-brain (configured; loaded=yes; healthy=yes)" in summary
    assert "coding: april-coding (configured; loaded=no; healthy=unknown)" in summary
    assert "reading: april-reading (configured; loaded=no; healthy=no)" in summary


def test_runtime_model_lifecycle_states_do_not_overclaim_loaded_state(settings_tmp) -> None:
    registry = ModelRegistry.from_file(
        project_root() / "configs" / "models.yaml", root=project_root()
    )
    summary = trusted_capability_summary(
        settings=settings_tmp,
        agent_registry=default_agent_registry(),
        tool_registry=default_registry(),
        model_registry=registry,
        runtime_evidence={
            "models": [
                {"id": "april-brain", "state": "unavailable"},
                {"id": "april-coding", "state": "loading"},
                {"id": "april-reading", "state": "unloading"},
            ],
            "health_status": "ok",
        },
    )
    assert "conversation/brain: april-brain (configured; loaded=no; healthy=unknown)" in summary
    assert "coding: april-coding (configured; loaded=unknown; healthy=unknown)" in summary
    assert "reading: april-reading (configured; loaded=unknown; healthy=unknown)" in summary


@pytest.mark.asyncio
async def test_runtime_self_evidence_malformed_or_failed_reads_are_non_fatal() -> None:
    class MalformedRuntime:
        async def models(self) -> object:
            return {"unexpected": []}

        async def health(self, *, timeout: float | None = None) -> dict[str, str]:
            del timeout
            return {"status": "ok"}

    class FailedRuntime:
        async def models(self) -> dict[str, list[object]]:
            raise RuntimeError("runtime offline")

        async def health(self, *, timeout: float | None = None) -> dict[str, str]:
            del timeout
            return {"status": "ok"}

    assert await collect_runtime_self_evidence(MalformedRuntime()) == {
        "models": [],
        "health_status": "ok",
    }
    assert await collect_runtime_self_evidence(FailedRuntime()) is None


def test_runtime_evidence_does_not_invent_loaded_model_state(settings_tmp) -> None:
    registry = ModelRegistry.from_file(
        project_root() / "configs" / "models.yaml", root=project_root()
    )
    summary = trusted_capability_summary(
        settings=settings_tmp,
        agent_registry=default_agent_registry(),
        tool_registry=default_registry(),
        model_registry=registry,
        runtime_evidence={"models": [], "health_status": "ok"},
    )
    assert "loaded=unknown; healthy=unknown (runtime did not report this model)" in summary
    assert "loaded=yes" not in summary
