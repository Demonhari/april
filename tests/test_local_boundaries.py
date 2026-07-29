from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest

from apps.cli.client import ApiOfflineError, AprilApiClient
from apps.daemon.apriald import daemon_lock_path, daemon_pid_path, daemon_status_path
from apps.runner import install as runner_install
from april_common.errors import PermissionDeniedError, RuntimeUnavailableError
from april_common.logging import JsonFormatter, configure_logging
from april_common.process_runner import (
    ProcessStatus,
    ResourceLimitProfile,
    ResourceLimitReport,
    RestrictedProcessResult,
)
from april_common.settings import AprilSettings
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


def test_v2_runtime_artifacts_are_ignored_in_scratch_checkout(tmp_path: Path) -> None:
    root = Path.cwd()
    settings = AprilSettings(home=root)
    gitignore_text = (root / ".gitignore").read_text(encoding="utf-8")
    entries = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    runtime_paths = {
        "data/evolution/",
        "data/playbooks/",
        "data/wake.sock",
        "data/voice.mute",
        "data/apriald.lock",
        "data/apriald.pid",
        "data/apriald.status.json",
    }
    derived_runtime_paths = {
        f"{settings.evolution_path.relative_to(root).as_posix()}/",
        f"{settings.playbooks_path.relative_to(root).as_posix()}/",
        settings.wake_socket_path.relative_to(root).as_posix(),
        settings.mute_flag_path.relative_to(root).as_posix(),
        daemon_lock_path(settings).relative_to(root).as_posix(),
        daemon_pid_path(settings).relative_to(root).as_posix(),
        daemon_status_path(settings).relative_to(root).as_posix(),
    }
    assert derived_runtime_paths == runtime_paths
    assert runtime_paths <= entries

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".gitignore").write_text(gitignore_text, encoding="utf-8")

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init", "--quiet")
    git("add", ".gitignore")
    git(
        "-c",
        "user.name=APRIL Tests",
        "-c",
        "user.email=tests@localhost",
        "commit",
        "--quiet",
        "-m",
        "baseline",
    )
    for runtime_path in runtime_paths:
        artifact = checkout / runtime_path
        if runtime_path.endswith("/"):
            artifact.mkdir(parents=True)
            (artifact / "runtime-artifact").write_text("generated\n", encoding="utf-8")
        else:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("generated\n", encoding="utf-8")

    assert git("status", "--porcelain").stdout == ""


def test_runtime_import_graph_excludes_evolution_and_memory() -> None:
    script = """
import importlib
import sys

for name in (
    'services.april_runtime.model_registry',
    'services.april_runtime.model_loader',
    'services.april_runtime.model_lifecycle',
    'services.april_runtime.server',
):
    importlib.import_module(name)
blocked = sorted(
    name for name in sys.modules
    if name == 'services.evolution' or name.startswith('services.evolution.')
    or name == 'services.memory' or name.startswith('services.memory.')
)
if blocked:
    raise SystemExit(','.join(blocked))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_only_llama_cpp_backend_imports_llama_cpp() -> None:
    root = Path.cwd()
    allowed = Path("services/april_runtime/llama_cpp_backend.py")
    violations: list[str] = []
    for package in ("april_common", "apps", "services", "agents", "skills"):
        for path in (root / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports_llama_cpp = any(
                (
                    isinstance(node, ast.Import)
                    and any(alias.name == "llama_cpp" for alias in node.names)
                )
                or (isinstance(node, ast.ImportFrom) and node.module == "llama_cpp")
                for node in ast.walk(tree)
            )
            relative = path.relative_to(root)
            if imports_llama_cpp and relative != allowed:
                violations.append(str(relative))
    assert violations == []


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
    posted: ClassVar[list[tuple[str, dict[str, Any]]]] = []

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
        self.posted.append((url, dict(json)))
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
    FakeRuntimeAsyncClient.posted = []
    client = RuntimeClient("http://127.0.0.1:2", generation_thread_provider=lambda: 6)
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
    chat_payload = next(
        payload for url, payload in FakeRuntimeAsyncClient.posted if url.endswith("/runtime/chat")
    )
    load_payload = next(
        payload
        for url, payload in FakeRuntimeAsyncClient.posted
        if url.endswith("/runtime/models/load")
    )
    assert chat_payload["generation_threads"] == 6
    assert load_payload["generation_threads"] == 6


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
        "delete old logs": "log_cleanup",
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

    def process_result(
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> RestrictedProcessResult:
        return RestrictedProcessResult(
            status=ProcessStatus.COMPLETED,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=False,
            stderr_truncated=False,
            duration_seconds=0.01,
            failure_code=None,
            resource_limits=ResourceLimitReport(ResourceLimitProfile.MODEL_UTILITY, (), ()),
        )

    async def fake_exec(argv: list[str], **kwargs: object) -> RestrictedProcessResult:
        return process_result(stdout="transcript")

    monkeypatch.setattr("services.voice.speech_to_text.run_restricted_process", fake_exec)
    stt = WhisperCppSpeechToText(binary, model)
    assert await stt.transcribe(audio) == "transcript"

    async def fake_piper(argv: list[str], **kwargs: object) -> RestrictedProcessResult:
        output_path = Path(argv[argv.index("--output_file") + 1])
        output_path.write_bytes(b"RIFFfake")
        return process_result()

    monkeypatch.setattr("services.voice.text_to_speech.run_restricted_process", fake_piper)
    tts = PiperTextToSpeech(binary, model)
    assert await tts.synthesize("hello", tmp_path / "out.wav") == tmp_path / "out.wav"

    async def silent_piper(argv: list[str], **kwargs: object) -> RestrictedProcessResult:
        return process_result()

    monkeypatch.setattr("services.voice.text_to_speech.run_restricted_process", silent_piper)
    with pytest.raises(RuntimeUnavailableError, match="non-empty WAV"):
        await tts.synthesize("hello", tmp_path / "missing.wav")

    async def failing_exec(argv: list[str], **kwargs: object) -> RestrictedProcessResult:
        return process_result(returncode=1, stderr="bad")

    monkeypatch.setattr("services.voice.speech_to_text.run_restricted_process", failing_exec)
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
    # The configured GGUF path does not exist, but a fake runtime is still ok and
    # explicitly marked simulated; the missing model stays informational.
    assert health.status == "ok"
    assert health.simulated is True
    assert health.missing_models == ["brain"]
    assert health.process_rss_bytes == 1234
    assert health.process_peak_rss_bytes == 4096
    assert health.process_memory_estimated is False
    assert health.models[0].load_duration_ms is not None
    assert health.models[0].threads == 1
    assert health.models[0].n_batch == 32
    assert await loader.unload("brain")


def _missing_model_registry(home: Path, backend: str) -> ModelRegistry:
    return ModelRegistry.from_dict(
        {
            "models": {
                "brain": {
                    "id": "brain",
                    "name": "Brain",
                    "path": "models/missing.gguf",
                    "backend": backend,
                    "role": "brain",
                    "chat_format": "generic",
                    "threads": 1,
                    "context_size": 512,
                    "temperature": 0.0,
                    "max_output_tokens": 16,
                    "keep_loaded": False,
                }
            }
        },
        root=home,
    )


def _no_metrics() -> ProcessMemoryMetrics:
    return ProcessMemoryMetrics(rss_bytes=None, peak_rss_bytes=None, estimated=True)


def test_runtime_health_simulated_is_honest_about_missing_models(settings_tmp) -> None:
    # Fake backend with a missing GGUF path: still ok, explicitly simulated, and
    # the missing model is reported as informational rather than hidden.
    fake = runtime_health(
        ModelLifecycle(_missing_model_registry(settings_tmp.home, "fake"), root_backend="fake"),
        backend="fake",
        metric_provider=_no_metrics,
    )
    assert fake.status == "ok"
    assert fake.simulated is True
    assert fake.missing_models == ["brain"]

    # Real backend with the same missing path: degraded and not simulated, so a
    # simulated run can never be mistaken for real-model readiness.
    real = runtime_health(
        ModelLifecycle(
            _missing_model_registry(settings_tmp.home, "llama_cpp"), root_backend="llama_cpp"
        ),
        backend="llama_cpp",
        metric_provider=_no_metrics,
    )
    assert real.status == "degraded"
    assert real.simulated is False
    assert real.missing_models == ["brain"]


def test_runtime_health_degrades_on_backend_error_even_when_simulated(settings_tmp) -> None:
    lifecycle = ModelLifecycle(
        _missing_model_registry(settings_tmp.home, "fake"), root_backend="fake"
    )
    # A genuine backend/model error must degrade health regardless of simulation.
    lifecycle.get_state("brain").state = "error"
    health = runtime_health(lifecycle, backend="fake", metric_provider=_no_metrics)
    assert health.status == "degraded"
    assert health.simulated is True
