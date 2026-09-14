from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import types
from pathlib import Path

import httpx
import pytest

from agents.schemas import AgentResult
from apps.runner.coding_compare import CodingCaseMeasurement, compare_coding_models
from april_common.errors import ModelUnavailableError, RuntimeUnavailableError
from april_common.process_runner import (
    ProcessStatus,
    ResourceLimitProfile,
    ResourceLimitReport,
    RestrictedProcessResult,
)
from services.april_runtime.colibri_backend import (
    ColibriBackend,
    _colibri_response_format,
    _content_text,
    _is_within,
    _json_object,
    _positive_int,
    colibri_manifest_digest,
    validate_colibri_url,
)
from services.april_runtime.model_lifecycle import ModelLifecycle
from services.april_runtime.model_registry import ModelDefinition, ModelRegistry
from services.april_runtime.schemas import ChatMessage, GenerationOptions, ResponseFormat
from services.brain.coding_workflow import CodingCompletionGate
from services.brain.context_memory import current_memory_context
from services.brain.delegation import DelegationCeiling, SpecialistTask
from services.brain.evidence import compact_tool_evidence
from services.brain.progress_controller import ProgressController, ProgressEvent
from services.brain.repository_state import RepositoryState, VerificationEvidence
from services.brain.run_controller import RunController
from services.brain.specialist_coordinator import SpecialistCoordinator
from services.brain.task_contract import (
    CapabilityCeiling,
    TaskContract,
    VerificationRequirements,
    constrain_task_contract,
    task_contract_for_request,
)
from services.evaluation.model_quality import ColibriEvaluationClient
from services.evolution.lessons import ExperienceLesson, LessonStore
from services.integrations.mcp.contracts import ExternalToolManifest, IntegrationEndpoint
from services.jobs import model_jobs
from services.jobs.model_jobs import validate_registered_model
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.state_facts import StateFact, StateFactKey, StateFactStore


def _colibri_model(root: Path) -> ModelDefinition:
    model_dir = root / "colibri-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    return ModelDefinition(
        id="coding-colibri",
        name="local coding model",
        path=model_dir,
        backend="colibri",
        artifact_kind="colibri_model_directory",
        colibri_base_url="http://127.0.0.1:4311",
        colibri_model_name="coding-local",
        colibri_expected_files=["config.json"],
        role="coding",
        threads=2,
        context_size=2048,
        temperature=0.2,
        max_output_tokens=64,
    )


@pytest.mark.asyncio
async def test_colibri_loopback_generation_and_privileged_tools_are_not_forwarded(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "coding-local"}]})
        if request.url.path == "/v1/chat/completions":
            payload = json.loads(request.content)
            assert "tools" not in payload
            assert "tool_choice" not in payload
            if payload.get("stream"):
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=(
                        b'data: {"choices":[{"delta":{"content":"st"}}]}\n\n'
                        b'data: {"choices":[{"delta":{"content":"ream"}}]}\n\n'
                        b"data: [DONE]\n\n"
                    ),
                )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "done"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                },
            )
        raise AssertionError(request.url.path)

    backend = ColibriBackend(
        base_url="http://127.0.0.1:4311",
        transport=httpx.MockTransport(transport),
        tokenizer=lambda text: list(range(len(text.split()))),
    )
    await backend.load(_colibri_model(tmp_path))
    assert await backend.tokenize("hello world") == [0, 1]
    result = await backend.generate("hello", temperature=0.1, max_output_tokens=8)
    assert result.text == "done"
    assert result.input_tokens == 3
    assert any(request.url.path == "/v1/chat/completions" for request in requests)
    streamed: list[str] = []
    async for chunk in backend.stream("hello", temperature=0.1, max_output_tokens=8):
        streamed.append(chunk)
    assert "".join(streamed) == "stream"
    await backend.unload()


def test_colibri_remote_endpoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="loopback"):
        validate_colibri_url("https://example.invalid/v1")
    with pytest.raises(ValueError, match="credentials"):
        validate_colibri_url("http://user:password@localhost:4311")
    with pytest.raises(ValueError, match="loopback"):
        ModelDefinition(
            id="remote",
            name="remote",
            path=Path("model"),
            backend="colibri",
            colibri_base_url="http://192.168.1.4:4311",
            role="coding",
            threads=1,
            context_size=256,
            temperature=0.2,
            max_output_tokens=16,
        )


@pytest.mark.asyncio
async def test_colibri_missing_endpoint_degrades_without_startup_corruption() -> None:
    backend = ColibriBackend()
    health = await backend.health()
    assert health.ok is False
    assert "configured but unavailable" in health.message


@pytest.mark.asyncio
async def test_colibri_malformed_health_and_model_identity_fail_closed(tmp_path: Path) -> None:
    def malformed(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, text="not-json")
        return httpx.Response(404)

    backend = ColibriBackend(
        base_url="http://127.0.0.1:4311", transport=httpx.MockTransport(malformed)
    )
    assert (await backend.health()).ok is False

    def mismatched(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "different"}]})
        return httpx.Response(404)

    backend = ColibriBackend(
        base_url="http://127.0.0.1:4311", transport=httpx.MockTransport(mismatched)
    )
    with pytest.raises(RuntimeUnavailableError, match="identity"):
        await backend.load(_colibri_model(tmp_path))


@pytest.mark.asyncio
async def test_colibri_artifact_tokenizer_and_malformed_responses_fail_closed(
    tmp_path: Path,
) -> None:
    model = _colibri_model(tmp_path)
    original_digest = colibri_manifest_digest(model, model.path)
    model.path.joinpath("config.json").write_text('{"revision": 2}', encoding="utf-8")
    assert colibri_manifest_digest(model, model.path) != original_digest

    def tokenizer_transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "coding-local"}]})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1, 2, 3]})
        raise AssertionError(request.url.path)

    backend = ColibriBackend(
        base_url=model.colibri_base_url,
        model_name=model.colibri_model_name,
        tokenizer=lambda text: list(range(len(text.split()))),
        transport=httpx.MockTransport(tokenizer_transport),
    )
    await backend.load(model)
    assert await backend.tokenize("one two three") == [0, 1, 2]
    assert backend.capabilities()["native_tools_forwarded"] is False
    await backend.unload()

    invalid_model = model.model_copy(update={"colibri_expected_files": ["missing.json"]})
    with pytest.raises(RuntimeUnavailableError, match="metadata"):
        await ColibriBackend().load(invalid_model)

    malformed_backend = ColibriBackend(
        base_url=model.colibri_base_url,
        model_name=model.colibri_model_name,
        tokenizer_path=tmp_path / "missing-tokenizer.json",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=(
                    {"status": "ready"}
                    if request.url.path == "/health"
                    else {"data": [{"id": "coding-local"}]}
                    if request.url.path == "/v1/models"
                    else {"tokens": "not-a-list"}
                ),
            )
        ),
    )
    with pytest.raises(RuntimeUnavailableError, match="exact local tokenizer"):
        await malformed_backend.load(model)


@pytest.mark.asyncio
async def test_colibri_missing_service_and_malformed_generation_are_unavailable(
    tmp_path: Path,
) -> None:
    model = _colibri_model(tmp_path)
    missing_endpoint = model.model_copy(update={"colibri_base_url": None})
    with pytest.raises(RuntimeUnavailableError, match="endpoint"):
        await ColibriBackend().load(missing_endpoint)

    def malformed(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "coding-local"}]})
        return httpx.Response(200, json={"choices": []})

    backend = ColibriBackend(
        base_url=model.colibri_base_url,
        model_name=model.colibri_model_name,
        tokenizer=lambda text: list(range(len(text.split()))),
        transport=httpx.MockTransport(malformed),
    )
    await backend.load(model)
    with pytest.raises(RuntimeUnavailableError, match="generation"):
        await backend.generate("hello", temperature=0.1, max_output_tokens=8)


@pytest.mark.asyncio
async def test_colibri_load_validation_and_stream_errors_are_safe(tmp_path: Path) -> None:
    model = _colibri_model(tmp_path)
    with pytest.raises(RuntimeUnavailableError, match="directory"):
        await ColibriBackend().load(model.model_copy(update={"path": Path("missing-colibri")}))
    missing_metadata = model.model_copy(update={"colibri_expected_files": ["missing.json"]})
    assert colibri_manifest_digest(missing_metadata, model.path)
    with pytest.raises(RuntimeUnavailableError, match="non-Colibri"):
        await ColibriBackend().load(model.model_copy(update={"backend": "fake"}))
    with pytest.raises(RuntimeUnavailableError, match="directory artifact"):
        await ColibriBackend().load(model.model_copy(update={"artifact_kind": "gguf_file"}))
    with pytest.raises(RuntimeUnavailableError, match="directory"):
        await ColibriBackend().load(model.model_copy(update={"path": tmp_path / "absent"}))

    def not_ready(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "starting"})
        raise AssertionError(request.url.path)

    with pytest.raises(RuntimeUnavailableError, match="unavailable"):
        await ColibriBackend(
            transport=httpx.MockTransport(not_ready), base_url=model.colibri_base_url
        ).load(model)

    def malformed_identity(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": "malformed"})
        raise AssertionError(request.url.path)

    with pytest.raises(RuntimeUnavailableError, match="identity"):
        await ColibriBackend(
            transport=httpx.MockTransport(malformed_identity),
            base_url=model.colibri_base_url,
            tokenizer_path=tmp_path / "missing-tokenizer.json",
        ).load(model)

    def bad_stream(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "coding-local"}]})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, content=b"data: not-json\n\n")
        raise AssertionError(request.url.path)

    backend = ColibriBackend(
        transport=httpx.MockTransport(bad_stream),
        base_url=model.colibri_base_url,
        model_name=model.colibri_model_name,
        tokenizer=lambda text: list(range(len(text.split()))),
    )
    await backend.load(model)
    with pytest.raises(RuntimeUnavailableError, match="streaming JSON"):
        async for _chunk in backend.stream("hello", temperature=0.1, max_output_tokens=8):
            pass


def test_colibri_helper_boundaries_are_typed_and_local(tmp_path: Path) -> None:
    model = _colibri_model(tmp_path).model_copy(
        update={"colibri_expected_files": ["../outside.json"]}
    )
    assert colibri_manifest_digest(model, tmp_path / "colibri-model")
    assert _content_text([{"text": "a"}, {"text": "b"}]) == "ab"
    assert _content_text(42) == ""
    assert _positive_int(3, 9) == 3
    assert _positive_int(-1, 9) == 9
    assert _positive_int("3", 9) == 9
    assert _is_within(tmp_path / "child", tmp_path)
    assert not _is_within(tmp_path.parent / "outside", tmp_path)
    with pytest.raises(RuntimeUnavailableError, match="invalid JSON"):
        _json_object(httpx.Response(200, text="not-json"))
    with pytest.raises(RuntimeUnavailableError, match="malformed JSON object"):
        _json_object(httpx.Response(200, json=[]))


@pytest.mark.asyncio
async def test_colibri_http_error_status_and_local_tokenizer_fallback(tmp_path: Path) -> None:
    model = _colibri_model(tmp_path)
    tokenizer_file = model.path / "tokenizer.json"
    tokenizer_file.write_text("{}", encoding="utf-8")

    def error_transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "coding-local"}]})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(500, json={"error": "failed"})
        raise AssertionError(request.url.path)

    backend = ColibriBackend(
        base_url=model.colibri_base_url,
        model_name=model.colibri_model_name,
        tokenizer=lambda text: list(range(len(text.split()))),
        transport=httpx.MockTransport(error_transport),
    )
    await backend.load(model)
    with pytest.raises(RuntimeUnavailableError, match="error"):
        await backend.generate("hello", temperature=0.1, max_output_tokens=8)
    with pytest.raises(RuntimeUnavailableError, match="streaming request"):
        async for _chunk in backend.stream("hello", temperature=0.1, max_output_tokens=8):
            pass
    await backend.unload()

    backend = ColibriBackend()
    with pytest.raises(RuntimeUnavailableError, match="endpoint"):
        await backend._http_client()
    assert await backend._validate_model_name() is None


@pytest.mark.asyncio
async def test_colibri_transport_and_optional_tokenizer_failures_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _colibri_model(tmp_path)

    def failing_transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "coding-local"}]})
        raise httpx.ConnectError("local service stopped", request=request)

    backend = ColibriBackend(
        base_url=model.colibri_base_url,
        model_name=model.colibri_model_name,
        tokenizer=lambda text: list(range(len(text.split()))),
        transport=httpx.MockTransport(failing_transport),
    )
    await backend.load(model)
    with pytest.raises(RuntimeUnavailableError, match="unavailable"):
        await backend.generate("hello", temperature=0.1, max_output_tokens=8)
    with pytest.raises(RuntimeUnavailableError, match="unavailable"):
        async for _chunk in backend.stream("hello", temperature=0.1, max_output_tokens=8):
            pass

    class FakeTokenizer:
        @classmethod
        def from_file(cls, _path: str) -> object:
            return cls()

        def encode(self, text: str) -> list[int]:
            return list(range(len(text.split())))

    monkeypatch.setitem(sys.modules, "tokenizers", types.SimpleNamespace(Tokenizer=FakeTokenizer))
    tokenizer_path = model.path / "local-tokenizer.json"
    tokenizer_path.write_text("{}", encoding="utf-8")
    local_backend = ColibriBackend(tokenizer_path=tokenizer_path)
    local_backend._load_optional_tokenizer()
    assert await local_backend.tokenize("one two") == [0, 1]

    class BrokenTokenizer:
        @classmethod
        def from_file(cls, _path: str) -> object:
            raise ValueError("invalid tokenizer")

    monkeypatch.setitem(sys.modules, "tokenizers", types.SimpleNamespace(Tokenizer=BrokenTokenizer))
    broken_backend = ColibriBackend(tokenizer_path=tokenizer_path)
    broken_backend._load_optional_tokenizer()
    assert broken_backend.capabilities()["tokenizer"] is False


@pytest.mark.asyncio
async def test_colibri_wire_contract_maps_thinking_and_structured_formats_without_seed(
    tmp_path: Path,
) -> None:
    model = _colibri_model(tmp_path)
    payloads: list[dict[str, object]] = []

    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "coding-local"}]})
        if request.url.path == "/v1/chat/completions":
            payloads.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "{}"}}]},
            )
        raise AssertionError(request.url.path)

    backend = ColibriBackend(
        base_url=model.colibri_base_url,
        model_name=model.colibri_model_name,
        tokenizer=lambda text: list(range(len(text.split()))),
        transport=httpx.MockTransport(transport),
    )
    await backend.load(model)
    messages = [ChatMessage(role="user", content="return json")]
    schema = ResponseFormat(type="json_object", json_schema={"type": "object"})
    await backend.generate_messages(
        "",
        messages=messages,
        temperature=0.0,
        max_output_tokens=8,
        seed=42,
        response_format=schema,
        disable_thinking=False,
    )
    await backend.generate_messages(
        "",
        messages=messages,
        temperature=0.0,
        max_output_tokens=8,
        response_format=ResponseFormat(type="json_object"),
        disable_thinking=True,
    )
    await backend.generate_messages(
        "",
        messages=messages,
        temperature=0.0,
        max_output_tokens=8,
    )
    assert payloads[0]["enable_thinking"] is True
    assert "seed" not in payloads[0]
    assert payloads[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"schema": {"type": "object"}},
    }
    assert payloads[1]["enable_thinking"] is False
    assert payloads[1]["response_format"] == {"type": "json_object"}
    assert "enable_thinking" not in payloads[2]
    assert backend.capabilities()["per_request_seed"] is False
    await backend.unload()


@pytest.mark.asyncio
async def test_colibri_does_not_assume_remote_tokenize_and_rejects_incomplete_sse(
    tmp_path: Path,
) -> None:
    model = _colibri_model(tmp_path)

    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "coding-local"}]})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
            )
        raise AssertionError(request.url.path)

    backend = ColibriBackend(
        base_url=model.colibri_base_url,
        model_name=model.colibri_model_name,
        tokenizer=lambda text: list(range(len(text.split()))),
        transport=httpx.MockTransport(transport),
    )
    await backend.load(model)
    with pytest.raises(RuntimeUnavailableError, match="not supported"):
        await ColibriBackend(base_url=model.colibri_base_url).tokenize("one")
    with pytest.raises(RuntimeUnavailableError, match="not supported"):
        await ColibriBackend(tokenizer_path=tmp_path / "missing-tokenizer.json").tokenize("one")
    with pytest.raises(RuntimeUnavailableError, match=r"before \[DONE\]"):
        async for _chunk in backend.stream("hello", temperature=0.0, max_output_tokens=8):
            pass
    await backend.unload()


@pytest.mark.asyncio
async def test_colibri_relative_tokenizer_usage_events_and_identity_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _colibri_model(tmp_path).model_copy(update={"colibri_tokenizer_path": "tokenizer.json"})
    (model.path / "tokenizer.json").write_text("{}", encoding="utf-8")

    class FakeTokenizer:
        @classmethod
        def from_file(cls, _path: str) -> object:
            return cls()

        def encode(self, text: str) -> list[int]:
            return list(range(len(text.split())))

    monkeypatch.setitem(sys.modules, "tokenizers", types.SimpleNamespace(Tokenizer=FakeTokenizer))

    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "coding-local"}]})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    b'data: {"usage":{"prompt_tokens":1}}\n\n'
                    b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                    b"data: [DONE]\n\n"
                ),
            )
        raise AssertionError(request.url.path)

    backend = ColibriBackend(
        base_url=model.colibri_base_url,
        model_name=model.colibri_model_name,
        transport=httpx.MockTransport(transport),
    )
    await backend.load(model)
    assert await backend.tokenize("one two") == [0, 1]
    backend._tokenizer = None
    assert await backend.tokenize("one two") == [0, 1]
    assert _colibri_response_format(ResponseFormat(type="text")) is None
    chunks = [
        chunk async for chunk in backend.stream("hello", temperature=0.0, max_output_tokens=8)
    ]
    assert chunks == ["ok"]
    await backend.unload()

    failing = ColibriBackend(base_url=model.colibri_base_url, model_name=model.colibri_model_name)

    async def unexpected(_method: str, _path: str, **_kwargs: object) -> httpx.Response:
        raise RuntimeError("unexpected local failure")

    failing._request = unexpected  # type: ignore[method-assign]
    with pytest.raises(RuntimeUnavailableError, match="could not be verified"):
        await failing._validate_model_name()


def test_colibri_local_protocol_helpers_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeUnavailableError, match="invalid JSON"):
        _json_object(httpx.Response(200, content=b"not-json"))
    with pytest.raises(RuntimeUnavailableError, match="malformed JSON object"):
        _json_object(httpx.Response(200, json=["not", "an", "object"]))
    missing_tokenizer = ColibriBackend(tokenizer_path=tmp_path / "missing-tokenizer.json")
    missing_tokenizer._load_optional_tokenizer()
    assert missing_tokenizer.capabilities()["tokenizer"] is False


def test_colibri_directory_is_not_used_as_resident_ram_estimate(tmp_path: Path) -> None:
    model = _colibri_model(tmp_path)
    assert model.projected_resident_gb(tmp_path) is None
    configured = model.model_copy(update={"resident_gb": 8.5})
    assert configured.projected_resident_gb(tmp_path) == 8.5


def test_colibri_registered_directory_is_validated_as_its_own_artifact_kind(settings_tmp) -> None:
    model_dir = settings_tmp.home / "models" / "coding-colibri"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (settings_tmp.home / "configs").mkdir(parents=True, exist_ok=True)
    (settings_tmp.home / "configs" / "models.yaml").write_text(
        """
models:
  coding:
    id: coding
    name: local coding
    path: models/coding-colibri
    backend: colibri
    artifact_kind: colibri_model_directory
    colibri_base_url: http://127.0.0.1:4311
    colibri_model_name: coding-local
    colibri_expected_files: [config.json]
    colibri_tokenizer_path: tokenizer.json
    resident_gb: 8.0
    role: coding
    threads: 2
    context_size: 2048
    temperature: 0.2
    max_output_tokens: 64
""",
        encoding="utf-8",
    )
    validated = validate_registered_model(settings_tmp, "coding")
    assert validated.artifact_kind == "colibri_model_directory"
    assert validated.size == 0
    assert validated.manifest_digest == validated.sha256


@pytest.mark.asyncio
async def test_colibri_load_without_resident_estimate_is_rejected_before_service_access(
    tmp_path: Path,
) -> None:
    model = _colibri_model(tmp_path)
    registry = ModelRegistry.from_dict(
        {"models": {"coding": model.model_dump(mode="json")}}, root=tmp_path
    )
    with pytest.raises(ModelUnavailableError, match="resident_gb"):
        await ModelLifecycle(registry).load_model(model.id)


def test_repository_state_capture_includes_changed_tracked_paths(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("before", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    tracked.write_text("after", encoding="utf-8")
    state = RepositoryState.capture(tmp_path, project_id="fixture")
    assert state.project_id == "fixture"
    assert state.affected_paths == ("tracked.txt",)


def test_repository_state_bound_completion_rejects_stale_evidence() -> None:
    state_a = RepositoryState("project", "head-a", "diff-a", "untracked-a")
    state_b = RepositoryState("project", "head-b", "diff-b", "untracked-b")
    evidence = VerificationEvidence(
        command_id="pytest",
        exit_status=0,
        result_status="pass",
        stdout_digest="stdout",
        stderr_digest="stderr",
        truncated=False,
        repository_state_digest=state_a.digest,
        test_targets=("tests/targeted.py",),
    )
    contract = TaskContract(
        run_id="run-1",
        user_goal="repair code",
        task_type="verified_code_modification",
        risk_class="write",
        verification=VerificationRequirements(required=True),
    )
    gate = CodingCompletionGate()
    gate.mark_stage("investigate")
    gate.mark_verification(evidence)
    assert gate.stage == "review"
    assert gate.can_complete(contract, state=state_a, evidence=evidence).allowed
    stale = gate.can_complete(contract, state=state_b, evidence=evidence)
    assert stale.allowed is False
    assert stale.reason == "verification_evidence_is_stale"


def test_coding_completion_gate_rejects_missing_or_failed_required_verification() -> None:
    state = RepositoryState("project", "head", "diff", "untracked")
    contract = TaskContract(
        run_id="run-1",
        user_goal="repair code",
        task_type="verified_code_modification",
        verification=VerificationRequirements(required=True),
    )
    gate = CodingCompletionGate()
    assert gate.can_complete(contract, state=state, evidence=None).reason == (
        "current_verification_required"
    )
    evidence = VerificationEvidence(
        command_id="pytest",
        exit_status=1,
        result_status="fail",
        stdout_digest="out",
        stderr_digest="err",
        truncated=True,
        repository_state_digest=state.digest,
        summary="three failures",
    )
    assert gate.can_complete(contract, state=state, evidence=evidence).reason == (
        "verification_failed"
    )
    assert (
        CodingCompletionGate()
        .can_complete(TaskContract(run_id="run-1", user_goal="explain"), state=state, evidence=None)
        .allowed
    )


def test_progress_controller_distinguishes_state_change_and_stops_repetition() -> None:
    controller = ProgressController(duplicate_warning_threshold=2, max_no_progress_cycles=3)
    event = ProgressEvent.from_values(
        action_name="pytest",
        normalized_arguments={"target": "unit"},
        input_state_digest="state-a",
        result_status="fail",
        evidence_digest="failure-a",
    )
    assert controller.observe(event).action == "record"
    assert controller.observe(event).action == "warn"
    assert controller.observe(event).action == "stop"
    changed = ProgressEvent.from_values(
        action_name="pytest",
        normalized_arguments={"target": "unit"},
        input_state_digest="state-b",
        result_status="fail",
        evidence_digest="failure-a",
    )
    assert controller.observe(changed).action == "record"
    fresh = ProgressEvent.from_values(
        action_name="pytest",
        normalized_arguments={"target": "unit"},
        input_state_digest="state-c",
        result_status="fail",
        evidence_digest="failure-b",
        new_evidence=True,
    )
    assert controller.observe(fresh).reason == "new_evidence"


@pytest.mark.asyncio
async def test_state_facts_are_project_scoped_and_late_observations_do_not_replace_newer(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "state.db")
    await database.connect()
    await run_migrations(database)
    try:
        store = StateFactStore(database)
        alpha = StateFact(
            key=StateFactKey(
                owner_scope="project", project_id="alpha", entity="backend", attribute="language"
            ),
            value="Python",
            evidence_reference="e-alpha",
            observed_at="2026-01-02T00:00:00Z",
            origin="user",
        )
        beta = alpha.model_copy(
            update={
                "id": "beta",
                "key": StateFactKey(
                    owner_scope="project", project_id="beta", entity="backend", attribute="language"
                ),
                "value": "Python",
                "evidence_reference": "e-beta",
            }
        )
        newer = alpha.model_copy(
            update={"id": "newer", "value": "Rust", "observed_at": "2026-01-03T00:00:00Z"}
        )
        older = alpha.model_copy(update={"id": "older", "value": "Go"})
        await store.put(alpha)
        await store.put(beta)
        await store.put(newer)
        await store.put(older)
        alpha_current = await store.current(alpha.key)
        beta_current = await store.current(beta.key)
        assert [fact.value for fact in alpha_current] == ["Rust"]
        assert [fact.value for fact in beta_current] == ["Python"]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_candidate_lesson_is_not_visible_until_approved(tmp_path: Path) -> None:
    database = Database(tmp_path / "lessons.db")
    await database.connect()
    await run_migrations(database)
    try:
        store = LessonStore(database)
        with pytest.raises(ValueError, match="candidates"):
            await store.create_candidate(
                ExperienceLesson(
                    project_id="alpha",
                    task_signature="invalid",
                    lesson="not yet a candidate",
                    status="approved",
                    creation_reason="test",
                )
            )
        lesson = await store.create_candidate(
            ExperienceLesson(
                project_id="alpha",
                task_signature="python-bug-fix",
                lesson="run the focused test after applying the patch",
                supporting_evidence_ids=("verification-1",),
                creation_reason="verified run",
            )
        )
        assert await store.approved(project_id="alpha") == []
        assert len(await store.list(status="candidate", project_id="alpha")) == 1
        assert (await store.get(lesson.id)) is not None
        assert len(await store.list(project_id="alpha")) == 1
        with pytest.raises(ValueError, match="candidate"):
            await store.transition(lesson.id, "candidate")
        await store.transition(lesson.id, "approved")
        assert (await store.approved(project_id="alpha"))[0].status == "approved"
        await store.transition(lesson.id, "superseded")
        assert (await store.get(lesson.id)).status == "superseded"
        assert await store.get("missing") is None
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_context_memory_includes_only_scoped_current_facts_and_approved_lessons(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "context-memory.db")
    await database.connect()
    await run_migrations(database)
    try:
        await StateFactStore(database).put(
            StateFact(
                key=StateFactKey(
                    owner_scope="project",
                    project_id="alpha",
                    entity="backend",
                    attribute="language",
                ),
                value="Tamil",
                evidence_reference="state-alpha",
                origin="user",
            )
        )
        lessons = LessonStore(database)
        candidate = await lessons.create_candidate(
            ExperienceLesson(
                project_id="alpha",
                task_signature="coding",
                lesson="candidate-only",
                creation_reason="test",
            )
        )
        approved = await lessons.create_candidate(
            ExperienceLesson(
                project_id="alpha",
                task_signature="coding",
                lesson="approved-only",
                creation_reason="test",
            )
        )
        await lessons.transition(approved.id, "approved")
        rendered = "\n".join(await current_memory_context(database, project_id="alpha"))
        assert "Tamil" in rendered
        assert "approved-only" in rendered
        assert "candidate-only" not in rendered
        truncated = await current_memory_context(database, project_id="alpha", max_chars=90)
        assert truncated[0].endswith("[TRUNCATED]")
        assert await current_memory_context(database, project_id="beta") == []
        assert (await lessons.get(candidate.id)).status == "candidate"
    finally:
        await database.close()


def test_delegation_can_only_reduce_authority() -> None:
    parent = DelegationCeiling(
        allowed_tools=frozenset({"repo_indexer", "test_runner", "patch_applier"}),
        maximum_risk_level=3,
        project_id="alpha",
        project_root="/project/alpha",
        maximum_depth=1,
    )
    specialist = DelegationCeiling(
        allowed_tools=frozenset({"repo_indexer", "test_runner"}),
        maximum_risk_level=1,
        project_id="alpha",
        project_root="/project/alpha",
        maximum_depth=1,
    )
    task = SpecialistTask.bounded(
        parent_run_id="run-1",
        role="reviewer",
        goal="inspect evidence",
        parent=parent,
        specialist=specialist,
    )
    assert task.capability_ceiling.allowed_tools == {"repo_indexer", "test_runner"}
    assert "patch_applier" not in task.capability_ceiling.allowed_tools
    assert task.capability_ceiling.maximum_risk_level == 1


def test_run_controller_bounds_replans_and_preserves_scope() -> None:
    contract = TaskContract(
        run_id="run-1",
        user_goal="repair code",
        maximum_replan_attempts=1,
    )
    controller = RunController.for_contract(contract)
    event = ProgressEvent.from_values(
        action_name="pytest",
        normalized_arguments={},
        input_state_digest="state",
        result_status="fail",
        evidence_digest="evidence",
    )
    assert controller.observe(event).action == "record"
    assert controller.request_replan() is True
    assert controller.request_replan() is False
    assert (
        "gather new evidence"
        in controller.trusted_feedback(
            ProgressController(duplicate_warning_threshold=1).observe(event)
        ).casefold()
    )
    with pytest.raises(ValueError, match="scope"):
        SpecialistTask.bounded(
            parent_run_id="run-1",
            role="implementer",
            goal="write code",
            parent=DelegationCeiling(project_id="alpha", project_root="/alpha"),
            specialist=DelegationCeiling(project_id="beta", project_root="/beta"),
        )
    with pytest.raises(ValueError, match="project"):
        SpecialistTask.bounded(
            parent_run_id="run-1",
            role="reviewer",
            goal="inspect",
            parent=DelegationCeiling(project_id="alpha", project_root="/same"),
            specialist=DelegationCeiling(project_id="beta", project_root="/same"),
        )
    with pytest.raises(ValueError, match="depth"):
        SpecialistTask.bounded(
            parent_run_id="run-1",
            role="researcher",
            goal="research",
            parent=DelegationCeiling(maximum_depth=0),
            specialist=DelegationCeiling(),
        )
    reduced = SpecialistTask.bounded(
        parent_run_id="run-1",
        role="investigator",
        goal="inspect",
        parent=DelegationCeiling(allowed_tools=frozenset({"read"}), maximum_risk_level=2),
        specialist=DelegationCeiling(
            allowed_tools=frozenset({"read", "write"}), maximum_risk_level=2
        ),
        global_policy=DelegationCeiling(allowed_tools=frozenset({"read"}), maximum_risk_level=1),
    )
    assert reduced.capability_ceiling.maximum_risk_level == 1
    assert (
        controller.completion_decision(
            state=RepositoryState("project", "head", "diff", "untracked"), evidence=None
        ).allowed
        is True
    )
    assert RunController.trusted_feedback(
        ProgressController(max_no_progress_cycles=1).observe(event)
    )


def test_mcp_schema_drift_requires_new_approval_and_remote_is_rejected() -> None:
    with pytest.raises(ValueError, match="local"):
        IntegrationEndpoint(kind="mcp_loopback", url="https://remote.invalid")
    base = ExternalToolManifest(
        integration_id="local",
        server_identity="server-a",
        tool_name="read",
        input_schema={"type": "object"},
        declared_risk=1,
        endpoint=IntegrationEndpoint(kind="mcp_stdio", command=("provider",)),
        specification_digest="spec-a",
        approval_id="approval-a",
    )
    changed = base.model_copy(update={"input_schema": {"type": "object", "properties": {}}})
    assert changed.schema_changed_from(base)
    with pytest.raises(ValueError, match="schema drift"):
        changed.activate_against(base, approval_id="approval-b")

    approved = (
        base.with_status("candidate")
        .with_status("validated")
        .with_status("verified")
        .with_status("approved")
    )
    active = approved.activate_against(approved, approval_id="approval-b")
    assert active.status == "active"
    with pytest.raises(ValueError, match="transition"):
        base.with_status("active", approval_id="approval-a")


def test_compacted_tool_evidence_retains_machine_fields_and_truncation() -> None:
    compact = compact_tool_evidence(
        action="pytest",
        argument_digest="args",
        repository_state_digest="state",
        exit_status=1,
        result_status="fail",
        output="important failure details\n" + ("verbose " * 100),
        max_important_chars=32,
    )
    assert compact.action == "pytest"
    assert compact.result_status == "fail"
    assert compact.truncated is True
    assert compact.output_digest
    plain = compact_tool_evidence(
        action="read_file",
        argument_digest="args",
        repository_state_digest=None,
        exit_status=0,
        result_status="pass",
        output="warning " * 20,
        max_important_chars=8,
    )
    assert plain.truncated is True


def test_coding_comparison_uses_identical_fixtures_without_activation() -> None:
    models = tuple(
        ModelDefinition(
            id=model_id,
            name=model_id,
            path=Path("unused"),
            backend="fake",
            artifact_kind="none",
            role="coding",
            threads=1,
            context_size=512,
            temperature=0.2,
            max_output_tokens=32,
        )
        for model_id in ("qwen-candidate", "kat-candidate")
    )

    def evaluate(model: ModelDefinition, fixture: str) -> CodingCaseMeasurement:
        return CodingCaseMeasurement(
            completed=model.id == "kat-candidate",
            tests_passed=model.id == "kat-candidate",
            structured_json_valid=True,
            action_valid=True,
            turns=2,
            latency_seconds=0.1,
        )

    report = compare_coding_models(models, evaluate)
    assert report.recommendation == "kat-candidate"
    assert report.automatic_activation_performed is False
    assert len(report.fixture_set) == 6

    tied = compare_coding_models(
        models,
        lambda model, fixture: CodingCaseMeasurement(
            completed=True,
            tests_passed=True,
            structured_json_valid=True,
            action_valid=True,
            turns=1,
            latency_seconds=0.1,
        ),
    )
    assert "candidate test pass rates are tied" in tied.warnings


def test_task_contract_derivation_and_caller_narrowing_cannot_expand_authority() -> None:
    derived = task_contract_for_request(
        run_id="run-1",
        user_goal="repair code",
        agent_name="coding_agent",
        agent_tools={"read_file", "test_runner", "patch_applier"},
        project_id="alpha",
        project_root="/project/alpha",
        intent="code_modification",
        permission_level=3,
        risk_level="code_write",
        allowed_scope=("/project/alpha",),
    )
    narrowed = TaskContract(
        run_id="run-1",
        user_goal="repair code",
        project_id="alpha",
        project_root="/project/alpha",
        task_type="verified_code_modification",
        risk_class="write",
        verification=VerificationRequirements(required=True),
        capability_ceiling=CapabilityCeiling(
            allowed_tools=frozenset({"read_file"}),
            maximum_risk_level=1,
        ),
    )
    accepted = constrain_task_contract(derived, narrowed, request_id="run-1")
    assert accepted.capability_ceiling.allowed_tools == {"read_file"}
    assert accepted.project_root == "/project/alpha"
    widened = narrowed.model_copy(
        update={"capability_ceiling": CapabilityCeiling(allowed_tools=frozenset({"shell"}))}
    )
    with pytest.raises(ValueError, match="tools"):
        constrain_task_contract(derived, widened, request_id="run-1")


def test_task_contract_rejects_each_authority_expansion() -> None:
    derived = task_contract_for_request(
        run_id="run-1",
        user_goal="repair code",
        agent_name="coding_agent",
        agent_tools={"read_file", "test_runner", "patch_applier"},
        project_id="alpha",
        project_root="/project/alpha",
        intent="code_modification",
        permission_level=3,
        risk_level="code_write",
        allowed_scope=("/project/alpha",),
    )
    base = TaskContract(
        run_id="run-1",
        user_goal="repair code",
        project_id="alpha",
        project_root="/project/alpha",
        task_type="code_modification",
        verification=VerificationRequirements(required=True),
        capability_ceiling=CapabilityCeiling(
            allowed_tools=frozenset({"read_file"}),
            maximum_risk_level=1,
        ),
    )
    cases = (
        (base.model_copy(update={"run_id": "other"}), "identity"),
        (base.model_copy(update={"project_id": "beta"}), "project"),
        (base.model_copy(update={"project_root": "/project/beta"}), "root"),
        (
            base.model_copy(update={"capability_ceiling": CapabilityCeiling(maximum_risk_level=4)}),
            "risk",
        ),
        (
            base.model_copy(update={"verification": VerificationRequirements(required=False)}),
            "verification",
        ),
        (base.model_copy(update={"maximum_specialist_depth": 2}), "delegation"),
        (base.model_copy(update={"maximum_replan_attempts": 2}), "re-plan"),
    )
    for requested, message in cases:
        with pytest.raises(ValueError, match=message):
            constrain_task_contract(derived, requested, request_id="run-1")
    with pytest.raises(ValueError, match="task type"):
        constrain_task_contract(
            derived.model_copy(update={"task_type": "code_modification"}),
            base.model_copy(update={"task_type": "verified_code_modification"}),
            request_id="run-1",
        )


@pytest.mark.asyncio
async def test_specialist_coordinator_bounds_depth_and_findings() -> None:
    coordinator = SpecialistCoordinator(loop=object())  # type: ignore[arg-type]
    contract = TaskContract(
        run_id="run-1",
        user_goal="repair code",
        maximum_specialist_depth=0,
        capability_ceiling=CapabilityCeiling(allowed_tools=frozenset({"read_file"})),
    )
    assert (
        await coordinator.investigate(
            parent_run_id="run-1",
            parent_contract=contract,
            parent_agent=None,  # type: ignore[arg-type]
            goal="inspect",
            context=None,  # type: ignore[arg-type]
            request_id="request",
            history=[],
            context_sections=[],
        )
        is None
    )
    result = AgentResult(status="ok", final_message="x" * 20)
    assert SpecialistCoordinator.bounded_findings(result, max_chars=10).endswith("[TRUNCATED]")


def test_run_controller_rejects_corrupt_persisted_stage() -> None:
    controller = RunController.for_contract(TaskContract(run_id="run-1", user_goal="inspect"))
    snapshot = controller.snapshot()
    snapshot["completion_stage"] = "not-a-stage"
    with pytest.raises(ValueError, match="stage"):
        RunController.restore(snapshot)


@pytest.mark.asyncio
async def test_model_utility_job_enriches_bounded_worker_result(
    settings_tmp, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated = model_jobs.ValidatedRegisteredModel(
        model_id="april-brain",
        role="brain",
        path=settings_tmp.home / "model.gguf",
        basename="model.gguf",
        size=4,
        sha256="digest",
    )
    worker_result = RestrictedProcessResult(
        status=ProcessStatus.COMPLETED,
        returncode=0,
        stdout='{"model_id":"april-brain","passed":true}',
        stderr="",
        stdout_truncated=False,
        stderr_truncated=False,
        duration_seconds=0.1,
        failure_code=None,
        resource_limits=ResourceLimitReport(
            requested_profile=ResourceLimitProfile.MODEL_UTILITY,
            applied=("cpu",),
            unsupported=(),
        ),
    )

    async def fake_worker(*args, **kwargs):
        return worker_result

    monkeypatch.setattr(model_jobs, "validate_registered_model", lambda *args, **kwargs: validated)
    monkeypatch.setattr(model_jobs, "run_restricted_process", fake_worker)
    result = await model_jobs.run_model_utility_job(
        settings_tmp,
        model_id="april-brain",
        mode="verify",
        cancellation_event=asyncio.Event(),
        timeout_seconds=1.0,
    )
    assert result["passed"] is True
    assert result["model_sha256"] == "digest"
    assert result["resource_limits_applied"] == ["cpu"]


@pytest.mark.asyncio
async def test_colibri_evaluation_client_maps_runtime_response() -> None:
    captured: dict[str, object] = {}

    class Backend:
        async def generate_messages(self, prompt, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(text="{}", input_tokens=2, output_tokens=1)

    client = ColibriEvaluationClient(backend=Backend(), model_id="coding-colibri")  # type: ignore[arg-type]
    response = await client.chat(
        model_id="coding-colibri",
        messages=[ChatMessage(role="user", content="return json")],
        options=GenerationOptions(enable_thinking=False, temperature=0.0, max_output_tokens=8),
        response_format=ResponseFormat(type="json_object"),
        request_id="evaluation-request",
    )
    assert response.content == "{}"
    assert response.usage.total_tokens == 3
    assert captured["disable_thinking"] is True
    assert captured["response_format"] == ResponseFormat(type="json_object")
