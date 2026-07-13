from __future__ import annotations

import json
import shutil
from pathlib import Path

import anyio
import pytest
from fastapi.testclient import TestClient

from april_common.audit import AuditLogger
from april_common.time import utc_now_iso
from services.api.server import create_app
from services.april_runtime.backend import BackendHealth, GenerationResult, RuntimeBackend
from services.april_runtime.model_lifecycle import ModelLifecycle
from services.april_runtime.model_registry import ModelDefinition, ModelRegistry
from services.evolution.adapters import (
    AdapterLifecycleManager,
    active_adapter_path_from_pointer,
    read_adapter_pointer,
    sha256_file,
)
from services.evolution.write_guard import EvolutionWriteGuard
from services.memory.database import Database
from services.memory.migrations import run_migrations
from tests.test_core_api import auth, make_container


class RecordingBackend(RuntimeBackend):
    async def load(self, model: ModelDefinition) -> None:
        return None

    async def unload(self) -> None:
        return None

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        return GenerationResult(text="ok", input_tokens=1, output_tokens=1)

    async def stream(
        self,
        prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
        top_p: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ):
        yield "ok"

    async def tokenize(self, text: str) -> list[int]:
        return [1]

    async def health(self) -> BackendHealth:
        return BackendHealth(ok=True, message="ok")


async def _manager(settings) -> tuple[Database, AdapterLifecycleManager]:
    database = Database(settings.database_path)
    await database.connect()
    await run_migrations(database)
    manager = AdapterLifecycleManager(
        settings,
        database,
        audit=AuditLogger(settings.audit_path),
    )
    return database, manager


def _adapter(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _evidence(settings, model_id: str, adapter: Path, *, adapter_ppl: float = 9.0) -> Path:
    path = settings.evolution_path / "adapters" / "evidence" / f"{adapter.stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_type": "lora_perplexity",
                "model_id": model_id,
                "adapter_path": str(adapter),
                "adapter_sha256": sha256_file(adapter),
                "base_perplexity": 10.0,
                "adapter_perplexity": adapter_ppl,
                "heldout_dataset": "fixture-heldout.jsonl",
                "created_at": utc_now_iso(),
            }
        ),
        encoding="utf-8",
    )
    return path


def _report(settings, model_id: str, adapter: Path) -> Path:
    path = settings.evolution_path / "adapters" / "evidence" / f"{adapter.stem}-report.json"
    path.write_text(
        json.dumps(
            {
                "report_type": "multi_model",
                "generated_at": utc_now_iso(),
                "runtime_backend": "llama_cpp",
                "summary": "pass",
                "models": [
                    {
                        "model_id": model_id,
                        "load_success": True,
                        "adapter_path_basename": adapter.name,
                        "adapter_sha256": sha256_file(adapter),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.asyncio
async def test_adapter_rollback_flips_pointer_to_prior_version(settings_tmp) -> None:
    database, manager = await _manager(settings_tmp)
    try:
        first = _adapter(settings_tmp.home / "models" / "adapters" / "v1.gguf", b"one")
        second = _adapter(settings_tmp.home / "models" / "adapters" / "v2.gguf", b"two")
        first_result = await manager.activate(
            model_id="april-brain",
            adapter_path=first,
            evidence_path=_evidence(settings_tmp, "april-brain", first),
        )
        second_result = await manager.activate(
            model_id="april-brain",
            adapter_path=second,
            evidence_path=_evidence(settings_tmp, "april-brain", second),
        )
        assert first_result.status == "activated"
        assert second_result.active_version == 2

        rollback = await manager.rollback(model_id="april-brain")
        assert rollback.status == "rolled_back"
        assert rollback.active_version == 1
        pointer = read_adapter_pointer(settings_tmp.home, "april-brain")
        assert pointer is not None
        assert pointer["active_version"] == 1
        assert active_adapter_path_from_pointer(settings_tmp.home, "april-brain") == first

        rows = await database.fetchall(
            "SELECT id, status FROM model_adapters WHERE model_id = ? ORDER BY id",
            ("april-brain",),
        )
        assert [(row["id"], row["status"]) for row in rows] == [
            ("april-brain:1", "active"),
            ("april-brain:2", "inactive"),
        ]
        assert "adapter_activated" in settings_tmp.audit_path.read_text(encoding="utf-8")
        assert "adapter_rolled_back" in settings_tmp.audit_path.read_text(encoding="utf-8")
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_adapter_activation_gate_blocks_absent_and_worse_evidence(settings_tmp) -> None:
    database, manager = await _manager(settings_tmp)
    try:
        adapter = _adapter(settings_tmp.home / "models" / "adapters" / "bad.gguf", b"bad")
        missing = await manager.activate(
            model_id="april-brain",
            adapter_path=adapter,
            evidence_path=None,
        )
        assert missing.status == "blocked"
        assert missing.next_command is not None
        assert read_adapter_pointer(settings_tmp.home, "april-brain") is None

        worse = await manager.activate(
            model_id="april-brain",
            adapter_path=adapter,
            evidence_path=_evidence(settings_tmp, "april-brain", adapter, adapter_ppl=11.0),
        )
        assert worse.status == "blocked"
        assert "worse" in str(worse.reason)
        assert read_adapter_pointer(settings_tmp.home, "april-brain") is None
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_adapter_activation_blocks_paths_outside_allowed_roots(settings_tmp) -> None:
    database, manager = await _manager(settings_tmp)
    try:
        outside = _adapter(settings_tmp.home.parent / "outside-adapter.gguf", b"outside")
        evidence = _evidence(settings_tmp, "april-brain", outside)
        result = await manager.activate(
            model_id="april-brain",
            adapter_path=outside,
            evidence_path=evidence,
        )
        assert result.status == "blocked"
        assert "outside configured allowed roots" in (result.reason or "")
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_production_adapter_activation_requires_report(settings_tmp) -> None:
    production = settings_tmp.model_copy(update={"environment": "production"})
    database, manager = await _manager(production)
    try:
        adapter = _adapter(production.home / "models" / "adapters" / "prod.gguf", b"prod")
        evidence = _evidence(production, "april-brain", adapter)
        blocked = await manager.activate(
            model_id="april-brain",
            adapter_path=adapter,
            evidence_path=evidence,
        )
        assert blocked.status == "blocked"
        assert "verification report" in str(blocked.reason)
        assert blocked.next_command is not None
        assert read_adapter_pointer(production.home, "april-brain") is None

        activated = await manager.activate(
            model_id="april-brain",
            adapter_path=adapter,
            evidence_path=evidence,
            verification_report_path=_report(production, "april-brain", adapter),
        )
        assert activated.status == "activated"
        assert read_adapter_pointer(production.home, "april-brain") is not None
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_missing_pointer_adapter_file_fails_model_load_hard(settings_tmp) -> None:
    database, manager = await _manager(settings_tmp)
    try:
        base = settings_tmp.home / "brain.gguf"
        base.write_bytes(b"GGUF")
        adapter = _adapter(settings_tmp.home / "models" / "adapters" / "gone.gguf", b"gone")
        result = await manager.activate(
            model_id="april-brain",
            adapter_path=adapter,
            evidence_path=_evidence(settings_tmp, "april-brain", adapter),
        )
        assert result.status == "activated"
        adapter.unlink()
        registry = ModelRegistry.from_dict(
            {
                "models": {
                    "brain": {
                        "id": "april-brain",
                        "name": "brain",
                        "path": str(base),
                        "backend": "llama_cpp",
                        "role": "brain",
                        "threads": 1,
                        "context_size": 512,
                        "temperature": 0.0,
                        "max_output_tokens": 16,
                    }
                }
            },
            root=settings_tmp.home,
        )
        lifecycle = ModelLifecycle(registry, backend_factory=lambda model: RecordingBackend())
        with pytest.raises(Exception, match="adapter path is missing"):
            await lifecycle.load_model("april-brain")
    finally:
        await database.close()


def test_adapter_pointer_fence_and_data_evolution_deletion(settings_tmp) -> None:
    adapter = _adapter(settings_tmp.home / "models" / "adapters" / "base.gguf", b"ok")
    pointer = {
        "schema_version": 1,
        "model_id": "april-brain",
        "created_at": "2026-07-13T00:00:00Z",
        "updated_at": "2026-07-13T00:00:00Z",
        "active_version": 1,
        "sha256": sha256_file(adapter),
        "versions": [
            {
                "version": 1,
                "adapter_path": str(adapter),
                "sha256": sha256_file(adapter),
                "created_at": "2026-07-13T00:00:00Z",
            }
        ],
    }
    guard = EvolutionWriteGuard(settings_tmp)
    guard.write_text(
        settings_tmp.evolution_path / "adapters" / "april-brain.json",
        json.dumps(pointer),
    )
    model = ModelDefinition.model_validate(
        {
            "id": "april-brain",
            "name": "brain",
            "path": "brain.gguf",
            "backend": "llama_cpp",
            "role": "brain",
            "threads": 1,
            "context_size": 512,
            "temperature": 0.0,
            "max_output_tokens": 16,
        }
    )
    assert model.resolved_adapter_path(settings_tmp.home) == adapter
    with pytest.raises(PermissionError):
        guard.write_text(settings_tmp.home / "configs" / "should-not-write.json", "{}")
    shutil.rmtree(settings_tmp.evolution_path)
    assert model.resolved_adapter_path(settings_tmp.home) is None


def test_adapter_api_activate_list_and_rollback(settings_tmp) -> None:
    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    adapter = _adapter(settings_tmp.home / "models" / "adapters" / "api.gguf", b"api")
    evidence = _evidence(settings_tmp, "april-brain", adapter)
    activated = client.post(
        "/evolution/adapters/activate",
        json={
            "model_id": "april-brain",
            "adapter_path": str(adapter),
            "evidence_path": str(evidence),
        },
        headers=auth(settings_tmp),
    )
    assert activated.status_code == 200
    assert activated.json()["activation"]["status"] == "activated"
    listed = client.get("/evolution/adapters", headers=auth(settings_tmp))
    assert listed.status_code == 200
    assert listed.json()["adapters"][0]["model_id"] == "april-brain"
    rollback = client.post(
        "/evolution/adapters/rollback",
        json={"model_id": "april-brain"},
        headers=auth(settings_tmp),
    )
    assert rollback.status_code == 200
    assert rollback.json()["rollback"]["status"] == "blocked"
