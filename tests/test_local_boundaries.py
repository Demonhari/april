from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from apps.cli.client import ApiOfflineError, AprilApiClient
from apps.runner import install as runner_install
from april_common.errors import PermissionDeniedError, RuntimeUnavailableError
from april_common.logging import JsonFormatter, configure_logging
from services.april_runtime.client import RuntimeClient
from services.april_runtime.health import ProcessMemoryMetrics, runtime_health
from services.april_runtime.model_lifecycle import ModelLifecycle
from services.april_runtime.model_loader import ModelLoader
from services.april_runtime.model_registry import ModelRegistry
from services.april_runtime.schemas import ChatMessage
from services.brain.fallback_router import FallbackRouter
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.retriever import MemoryRetriever
from services.memory.sqlite_memory import SqliteMemory
from services.memory.vector_memory import VectorMemory
from services.memory.writer import MemoryWriter
from services.voice.speech_to_text import WhisperCppSpeechToText
from services.voice.text_to_speech import PiperTextToSpeech
from skills.base import path_args
from skills.filesystem.common import ignored, safe_regex
from skills.filesystem.list_files import list_files
from skills.notes.create_note import create_note
from skills.notes.search_notes import search_notes
from skills.policy import ToolPolicy
from skills.registry import default_registry
from skills.reminders.create_reminder import create_reminder
from skills.reminders.list_reminders import list_reminders


class FakeApiAsyncClient:
    def __init__(self, *, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> FakeApiAsyncClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        if url.endswith("/offline"):
            raise httpx.ConnectError("offline")
        return httpx.Response(200, json={"url": url, "params": params, "headers": headers})

    async def post(
        self, url: str, *, json: dict[str, Any], headers: dict[str, str] | None = None
    ) -> httpx.Response:
        if url.endswith("/error"):
            return httpx.Response(403, json={"error": {"message": "denied"}})
        return httpx.Response(200, json={"url": url, "json": json, "headers": headers})

    async def delete(self, url: str, *, headers: dict[str, str]) -> httpx.Response:
        return httpx.Response(200, json={"deleted": url, "headers": headers})


@pytest.mark.asyncio
async def test_cli_client_get_post_delete_and_errors(monkeypatch) -> None:
    monkeypatch.setattr("apps.cli.client.httpx.AsyncClient", FakeApiAsyncClient)
    client = AprilApiClient("http://127.0.0.1:1/", "token", timeout=3)
    assert client.headers == {"Authorization": "Bearer token"}
    assert (await client.get("/health", params={"q": "x"}, auth=False))["headers"] is None
    posted = await client.post("/chat", {"message": "hello"})
    assert posted["headers"] == {"Authorization": "Bearer token"}
    assert (await client.delete("/memory/1"))["deleted"].endswith("/memory/1")
    with pytest.raises(ApiOfflineError, match="denied"):
        await client.post("/error", {})
    with pytest.raises(ApiOfflineError, match="APRIL API is offline"):
        await client.get("/offline")


class FakeRuntimeStream:
    def __init__(self, *, status_code: int = 200) -> None:
        self.status_code = status_code

    async def __aenter__(self) -> FakeRuntimeStream:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def aiter_lines(self):  # type: ignore[no-untyped-def]
        yield "event: token"
        yield 'data: {"token":"ok"}'


class FakeRuntimeAsyncClient:
    def __init__(self, *, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> FakeRuntimeAsyncClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def get(self, url: str, *, headers: dict[str, str] | None = None) -> httpx.Response:
        self.last_headers = headers
        if url.endswith("/runtime/health"):
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"models": []})

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        self.last_headers = headers
        if url.endswith("/runtime/chat"):
            return httpx.Response(
                200,
                json={
                    "request_id": json["request_id"] or "runtime-request",
                    "model_id": json["model_id"],
                    "content": "ok",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            )
        return httpx.Response(
            200,
            json={
                "request_id": json.get("request_id") or "op-request",
                "model_id": json["model_id"],
                "state": "loaded" if url.endswith("/load") else "unloaded",
                "message": "ok",
            },
        )

    def stream(
        self,
        method: str,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> FakeRuntimeStream:
        self.last_headers = headers
        return FakeRuntimeStream()


@pytest.mark.asyncio
async def test_runtime_client_methods_and_stream(monkeypatch) -> None:
    monkeypatch.setattr("services.april_runtime.client.httpx.AsyncClient", FakeRuntimeAsyncClient)
    client = RuntimeClient("http://127.0.0.1:2")
    response = await client.chat(
        model_id="april-brain",
        messages=[ChatMessage(role="user", content="hello")],
        request_id="request-1",
    )
    assert response.content == "ok"
    assert await client.models() == {"models": []}
    assert await client.health(timeout=0.1) == {"status": "ok"}
    assert (await client.load("april-brain")).state == "loaded"
    assert (await client.unload("april-brain")).state == "unloaded"
    assert [line async for line in client.stream(model_id="april-brain", messages=[])] == [
        '{"token":"ok"}'
    ]


def test_runner_install_main_uninstall_verify_and_shell_paths(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    bin_dir = tmp_path / "bin"
    home = tmp_path / "home"
    repo.mkdir()
    home.mkdir()
    assert (
        runner_install.main(["--install", "--repo-root", str(repo), "--bin-dir", str(bin_dir)]) == 0
    )
    assert runner_install.verify_wrappers(repo_root=repo, bin_dir=bin_dir) == []
    run_path = bin_dir / "run"
    run_path.write_text("broken", encoding="utf-8")
    errors = runner_install.verify_wrappers(repo_root=repo, bin_dir=bin_dir)
    assert any("required text" in error for error in errors)
    assert runner_install.shell_config_path(shell="/bin/zsh", home=home).name == ".zshrc"
    assert runner_install.shell_config_path(shell="/bin/bash", home=home).name == ".bashrc"
    with pytest.raises(ValueError, match="zsh and bash"):
        runner_install.shell_config_path(shell="/bin/fish", home=home)
    config_path, changed = runner_install.add_path_block(shell="/bin/zsh", home=home)
    assert changed is True
    assert runner_install.add_path_block(shell="/bin/zsh", home=home) == (config_path, False)
    monkeypatch.setenv("PATH", str(bin_dir))
    assert runner_install.path_contains_dir(bin_dir)
    assert runner_install.main(["--uninstall", "--bin-dir", str(bin_dir)]) == 0


def test_fallback_router_covers_local_intents() -> None:
    router = FallbackRouter()
    cases = {
        "delete old logs": "destructive_action",
        "please deploy this": "external_action",
        "apply the fix": "code_modification",
        "why is the repository animation broken": "coding_repo_analysis",
        "summarize this file": "document_reading",
        "remember my project preference": "memory_write",
        "remind me to stand up": "reminders",
        "write a story": "creative_writing",
        "plan today": "planning",
        "hello": "normal_conversation",
    }
    for message, intent in cases.items():
        assert router.route(message).intent == intent


def test_logging_formatter_includes_request_metadata() -> None:
    import logging

    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="april.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.request_id = "request-1"  # type: ignore[attr-defined]
    formatted = formatter.format(record)
    assert "request-1" in formatted
    assert "hello" in formatted
    configure_logging(logging.DEBUG)


def test_tool_policy_facade_uses_permission_engine(settings_tmp) -> None:
    policy = ToolPolicy(default_registry())
    decision = policy.evaluate(
        tool="read_file",
        args={"path": str(settings_tmp.home / "README.md")},
        agent="coding_agent",
    )
    assert decision.permission_level == 1


@pytest.mark.asyncio
async def test_memory_retriever_writer_and_skill_wrappers(settings_tmp, tmp_path: Path) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    vector = VectorMemory(settings_tmp.vector_index_path)
    retriever = MemoryRetriever(memory, vector)
    writer = MemoryWriter(memory)
    with pytest.raises(PermissionDeniedError):
        await writer.write("password is secret", reason="bad")
    durable = await writer.write("I prefer local models", reason="", requested_by_user=True)
    assert durable.content == "I prefer local models"
    await memory.create_memory("token should not appear", reason="sensitive")
    vector.index_chunks(
        source_type="repo",
        source_id="repo-1",
        project_id="project-1",
        chunks=[("README.md", "animation fix details", 1, 2)],
    )
    assert [result.content for result in await retriever.recent_memories(limit=5)]
    hybrid = await retriever.hybrid_search("local")
    assert all("token" not in result.content for result in hybrid)
    chunks = retriever.repo_chunks("animation", project_id="project-1", max_chars=10)
    assert chunks[0].content == "animation "
    note = await create_note({"title": "My Note", "content": "hello"})
    assert Path(note.data["path"]).exists()
    notes = await search_notes({"query": "hello"})
    assert notes.data["matches"]
    reminder = await create_reminder({"content": "stand up"})
    assert reminder.ok is True
    reminders = await list_reminders({})
    assert "stand up" in reminders.stdout
    await database.close()


@pytest.mark.asyncio
async def test_filesystem_helpers_and_list_files(settings_tmp, tmp_path: Path) -> None:
    root = settings_tmp.home
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("ignored", encoding="utf-8")
    (root / ".gitignore").write_text("*.tmp\n", encoding="utf-8")
    (root / "keep.txt").write_text("keep", encoding="utf-8")
    (root / "skip.tmp").write_text("skip", encoding="utf-8")
    result = await list_files({"path": str(root), "limit": 10})
    assert result.ok is True
    assert "keep.txt" in result.stdout
    assert "skip.tmp" not in result.stdout
    assert ignored(root / ".git" / "config", root=root, patterns=[])
    assert safe_regex("animation").search("Animation")
    with pytest.raises(ValueError, match="too long"):
        safe_regex("x" * 201)
    assert path_args({"path": "a", "repo_path": "b", "other": "c"}) == ["a", "b"]


class FakeProcess:
    def __init__(
        self, *, returncode: int = 0, stdout: bytes = b"text", stderr: bytes = b""
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.killed = False

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        return self.stdout, self.stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> None:
        return None


@pytest.mark.asyncio
async def test_voice_subprocess_adapters_success_and_failure(tmp_path: Path, monkeypatch) -> None:
    binary = tmp_path / "bin"
    model = tmp_path / "model"
    audio = tmp_path / "audio.wav"
    binary.write_text("", encoding="utf-8")
    model.write_text("", encoding="utf-8")
    audio.write_text("", encoding="utf-8")

    async def fake_exec(*argv: str, **kwargs: object) -> FakeProcess:
        return FakeProcess(stdout=b"transcript")

    monkeypatch.setattr("services.voice.speech_to_text.asyncio.create_subprocess_exec", fake_exec)
    stt = WhisperCppSpeechToText(binary, model)
    assert await stt.transcribe(audio) == "transcript"

    async def fake_piper(*argv: str, **kwargs: object) -> FakeProcess:
        return FakeProcess(stdout=b"", stderr=b"")

    monkeypatch.setattr("services.voice.text_to_speech.asyncio.create_subprocess_exec", fake_piper)
    tts = PiperTextToSpeech(binary, model)
    assert await tts.synthesize("hello", tmp_path / "out.wav") == tmp_path / "out.wav"

    async def failing_exec(*argv: str, **kwargs: object) -> FakeProcess:
        return FakeProcess(returncode=1, stderr=b"bad")

    monkeypatch.setattr(
        "services.voice.speech_to_text.asyncio.create_subprocess_exec", failing_exec
    )
    with pytest.raises(RuntimeUnavailableError, match="failed"):
        await stt.transcribe(audio)


@pytest.mark.asyncio
async def test_model_loader_and_runtime_health(settings_tmp) -> None:
    registry = ModelRegistry.from_dict(
        {
            "models": {
                "brain": {
                    "id": "brain",
                    "name": "Brain",
                    "path": "models/missing.gguf",
                    "backend": "fake",
                    "role": "brain",
                    "chat_format": "generic",
                    "threads": 1,
                    "n_batch": 32,
                    "context_size": 512,
                    "temperature": 0.0,
                    "max_output_tokens": 16,
                    "keep_loaded": True,
                }
            }
        },
        root=settings_tmp.home,
    )
    lifecycle = ModelLifecycle(registry, root_backend="fake")
    loader = ModelLoader(lifecycle)
    state = await loader.load("brain")
    assert state.state == "loaded"
    health = runtime_health(
        lifecycle,
        backend="fake",
        request_id="health-1",
        metric_provider=lambda: ProcessMemoryMetrics(
            rss_bytes=1234,
            peak_rss_bytes=4096,
            estimated=False,
        ),
    )
    assert health.request_id == "health-1"
    assert health.loaded_model_count == 1
    assert health.active_requests == 0
    assert health.process_rss_bytes == 1234
    assert health.process_peak_rss_bytes == 4096
    assert health.process_memory_estimated is False
    assert health.models[0].load_duration_ms is not None
    assert health.models[0].threads == 1
    assert health.models[0].n_batch == 32
    assert await loader.unload("brain")
