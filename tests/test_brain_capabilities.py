from __future__ import annotations

import asyncio

import pytest

from agents.base import USER_FACING_ASSISTANT_NAME, USER_FACING_IDENTITY_RULE
from agents.registry import default_agent_registry
from april_common.settings import project_root
from services.april_runtime.model_registry import ModelRegistry
from services.brain.capabilities import (
    collect_runtime_self_evidence,
    is_conversation_recall_request,
    is_self_introspection_request,
    is_voice_capability_request,
    render_conversation_recall_response,
    stable_prefix_prompt,
    trusted_capability_summary,
    trusted_capability_summary_parts,
    voice_capability_intent,
)
from services.brain.request_context import RequestContext, render_request_context
from services.memory.schemas import Message
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
    assert "state=unknown; loaded=unknown; healthy=unknown" in summary
    assert "april-brain (configured; loaded=yes" not in summary


@pytest.mark.asyncio
async def test_runtime_self_context_reports_only_evidenced_state() -> None:
    runtime = FakeRuntimeClient()
    evidence = await collect_runtime_self_evidence(runtime)
    assert evidence == {"models": [], "health_status": "ok", "backend": "fake"}
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
    assert (
        "conversation/brain: april-brain (configured; state=loaded; loaded=yes; healthy=yes)"
        in summary
    )
    assert (
        "coding: april-coding (configured; state=unloaded; loaded=no; healthy=unknown)" in summary
    )
    assert "reading: april-reading (configured; state=error; loaded=no; healthy=no)" in summary


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
    assert (
        "conversation/brain: april-brain (configured; state=unavailable; loaded=no; "
        "healthy=unknown)" in summary
    )
    assert (
        "coding: april-coding (configured; state=loading; loaded=unknown; healthy=unknown)"
        in summary
    )
    assert (
        "reading: april-reading (configured; state=unloading; loaded=unknown; "
        "healthy=unknown)" in summary
    )


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
    assert (
        "state=unknown; loaded=unknown; healthy=unknown (runtime did not report this model)"
        in summary
    )
    assert "loaded=yes" not in summary


def test_runtime_status_preserves_simulation_and_unknown_state(settings_tmp) -> None:
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
                {"id": "april-brain", "state": "bogus"},
                {"id": "april-coding", "state": "unloaded"},
            ],
            "health_status": "degraded",
            "backend": "fake",
            "simulated": True,
        },
    )
    assert "conversation/brain: april-brain (configured; state=unknown; loaded=unknown" in summary
    assert "coding: april-coding (configured; state=unloaded; loaded=no" in summary
    assert "Runtime backend evidence is simulated" in summary
    assert "bogus" not in summary
    assert "GGUF loading" in summary


def test_runtime_status_labels_failed_evidence_as_unavailable(settings_tmp) -> None:
    registry = ModelRegistry.from_file(
        project_root() / "configs" / "models.yaml", root=project_root()
    )
    summary = trusted_capability_summary(
        settings=settings_tmp,
        agent_registry=default_agent_registry(),
        tool_registry=default_registry(),
        model_registry=registry,
        runtime_evidence=None,
    )
    assert "Runtime evidence is unavailable" in summary
    assert "runtime status not queried" not in summary
    assert "state=unknown; loaded=unknown; healthy=unknown" in summary


def test_stable_prefix_layout_partitions_without_changing_default_summary(settings_tmp) -> None:
    registry = ModelRegistry.from_file(
        project_root() / "configs" / "models.yaml", root=project_root()
    )
    kwargs = {
        "settings": settings_tmp,
        "agent_registry": default_agent_registry(),
        "tool_registry": default_registry(),
        "model_registry": registry,
    }
    text_summary = trusted_capability_summary(
        **kwargs, request_context=RequestContext.from_origin("text", settings_tmp)
    )
    voice_summary = trusted_capability_summary(
        **kwargs, request_context=RequestContext.from_origin("voice", settings_tmp)
    )
    stable_text, volatile_text = trusted_capability_summary_parts(
        **kwargs, request_context=RequestContext.from_origin("text", settings_tmp)
    )
    stable_voice, volatile_voice = trusted_capability_summary_parts(
        **kwargs, request_context=RequestContext.from_origin("voice", settings_tmp)
    )
    stable_evidence, volatile_evidence = trusted_capability_summary_parts(
        **kwargs,
        runtime_evidence={"simulated": False, "models": {}},
        request_context=RequestContext.from_origin("voice", settings_tmp),
    )
    assert stable_text == stable_voice
    assert stable_voice == stable_evidence
    assert "Request origin: voice" not in stable_evidence
    assert "Voice interface configured:" in stable_evidence
    assert "Runtime evidence" in volatile_evidence
    assert set((stable_text + "\n" + volatile_text).splitlines()) == set(text_summary.splitlines())
    assert set((stable_voice + "\n" + volatile_voice).splitlines()) == set(
        voice_summary.splitlines()
    )
    assert stable_prefix_prompt(stable_text, volatile_text).startswith(
        "[APRIL_STABLE_PREFIX_LAYOUT]\n"
    )


def test_voice_request_context_is_transport_only_and_origin_scoped(settings_tmp) -> None:
    voice = RequestContext.from_origin("voice", settings_tmp)
    text = RequestContext.from_origin("text", settings_tmp)

    voice_lines = "\n".join(render_request_context(voice))
    text_lines = "\n".join(render_request_context(text))
    assert "Request origin: voice" in voice_lines
    assert "text transcript received at APRIL's voice endpoint" in voice_lines
    assert "does not prove physical capture" in voice_lines
    assert "Request origin: text" in text_lines
    assert "This request arrived as text" in text_lines
    assert voice.voice_enabled is False
    assert voice.voice_configured is False
    assert text.origin != voice.origin


def test_voice_capability_matcher_is_narrow_and_does_not_swallow_quoted_or_mixed_text() -> None:
    assert is_voice_capability_request("Can you hear me?") is True
    assert is_voice_capability_request("Do you have audio capabilities?") is True
    assert (
        voice_capability_intent("Does APRIL support voice input and spoken replies?") == "support"
    )
    assert (
        voice_capability_intent("This is a microphone test. Did you receive my message?")
        == "receipt"
    )
    assert is_conversation_recall_request("What did I just ask you to confirm?") is True
    assert is_conversation_recall_request("What did I ask you to confirm?") is True
    assert is_voice_capability_request('The quote says "Can you hear me?"') is False
    assert is_voice_capability_request("Can you hear me? Also delete a file.") is False
    assert is_conversation_recall_request('The document says "What did I just ask?"') is False


def test_conversation_recall_reports_only_available_history() -> None:
    earlier = Message(
        id="message-1",
        conversation_id="conversation-1",
        role="user",
        content="The earlier question was about local transcripts.",
        created_at="2026-01-01T00:00:00Z",
    )
    assert "local transcripts" in render_conversation_recall_response(
        [earlier], history_complete=True
    )
    assert "unavailable or truncated" in render_conversation_recall_response(
        [], history_complete=False
    )
    assert "won't infer one from durable memory" in render_conversation_recall_response(
        [], history_complete=True
    )
