from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path
from typing import Any

import pytest
import yaml

from services.jobs import model_import as model_import_module
from services.jobs.model_import import (
    ModelImportError,
    ModelImportService,
    reconcile_model_imports,
)


def _configure(settings_tmp: Any) -> ModelImportService:
    config = settings_tmp.home / "configs" / "models.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("models: {}\n", encoding="utf-8")
    return ModelImportService(settings_tmp)


def _gguf(path: Path, *, size: int = 2_500_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"x" * size)
    return path


async def _run(
    service: ModelImportService,
    source: Path,
    *,
    operation_id: str = "import-one",
    model_id: str = "candidate-one",
    expected_sha256: str | None = None,
    cancellation_event: asyncio.Event | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    payload = service.prepare_payload(
        source_path=str(source),
        model_id=model_id,
        role="brain",
        name="Candidate",
    )

    async def noop(_percent: int, _code: str) -> None:
        return None

    return await service.run(
        operation_id=operation_id,
        source_path=str(source),
        model_id=model_id,
        role="brain",
        name="Candidate",
        expected_sha256=expected_sha256 or str(payload["expected_sha256"]),
        cancellation_event=cancellation_event or asyncio.Event(),
        progress=progress or noop,
    )


@pytest.mark.asyncio
async def test_model_import_stages_reports_progress_and_registers_inactive(
    settings_tmp: Any,
) -> None:
    service = _configure(settings_tmp)
    source = _gguf(settings_tmp.home / "incoming" / "candidate.gguf")
    updates: list[tuple[int, str]] = []

    async def progress(percent: int, code: str) -> None:
        updates.append((percent, code))

    result = await _run(service, source, progress=progress)
    assert result == {
        "model_id": "candidate-one",
        "logical_role": "brain",
        "basename": "candidate.gguf",
        "byte_count": source.stat().st_size,
        "sha256": service.prepare_payload(
            source_path=str(source),
            model_id="candidate-one",
            role="brain",
            name="Candidate",
        )["expected_sha256"],
        "registration_status": "registered_inactive",
    }
    assert updates[0] == (5, "model_import_validated")
    assert updates[-1] == (100, "model_import_completed")
    assert any(code == "model_import_copying" for _percent, code in updates)
    registered = yaml.safe_load(
        (settings_tmp.home / "configs" / "models.yaml").read_text(encoding="utf-8")
    )["models"]["candidate-one"]
    assert registered["keep_loaded"] is False
    assert registered["priority"] == -100
    assert not list((settings_tmp.home / ".april_tmp" / "model-import").glob("*.part"))


@pytest.mark.asyncio
async def test_model_import_cancellation_removes_staging(settings_tmp: Any) -> None:
    service = _configure(settings_tmp)
    source = _gguf(settings_tmp.home / "incoming" / "cancel.gguf")
    cancellation = asyncio.Event()

    async def progress(_percent: int, code: str) -> None:
        if code == "model_import_copying":
            cancellation.set()

    with pytest.raises(asyncio.CancelledError):
        await _run(
            service,
            source,
            operation_id="cancel-import",
            cancellation_event=cancellation,
            progress=progress,
        )
    assert not (settings_tmp.home / "models" / source.name).exists()
    assert not list((settings_tmp.home / ".april_tmp" / "model-import").glob("*.part"))


@pytest.mark.asyncio
async def test_model_import_rejects_hash_magic_symlink_and_overwrite(
    settings_tmp: Any,
) -> None:
    service = _configure(settings_tmp)
    invalid = settings_tmp.home / "incoming" / "invalid.gguf"
    invalid.parent.mkdir()
    invalid.write_bytes(b"NOPE")
    with pytest.raises(ModelImportError, match="invalid_gguf_magic"):
        service.prepare_payload(
            source_path=str(invalid),
            model_id="invalid",
            role="brain",
            name="Invalid",
        )
    source = _gguf(settings_tmp.home / "incoming" / "valid.gguf", size=16)
    symlink = settings_tmp.home / "incoming" / "link.gguf"
    symlink.symlink_to(source)
    with pytest.raises(ModelImportError, match="symlink_rejected"):
        service.prepare_payload(
            source_path=str(symlink),
            model_id="linked",
            role="brain",
            name="Linked",
        )
    with pytest.raises(ModelImportError, match="hash_mismatch"):
        await _run(service, source, expected_sha256="0" * 64)
    (settings_tmp.home / "models").mkdir(exist_ok=True)
    (settings_tmp.home / "models" / source.name).write_bytes(b"existing")
    with pytest.raises(ModelImportError, match="overwrite_rejected"):
        await _run(service, source, operation_id="overwrite")


@pytest.mark.asyncio
async def test_model_import_config_failure_restores_previous_state(settings_tmp: Any) -> None:
    normal = _configure(settings_tmp)
    source = _gguf(settings_tmp.home / "incoming" / "rollback.gguf", size=64)
    config = settings_tmp.home / "configs" / "models.yaml"
    before = config.read_bytes()

    def fail_config(path: Path, payload: bytes) -> None:
        if path == config:
            raise OSError("injected config failure")
        model_import_module._atomic_write_bytes(path, payload)

    service = ModelImportService(settings_tmp, atomic_writer=fail_config)
    with pytest.raises(OSError, match="injected"):
        await _run(service, source, operation_id="rollback-import")
    assert config.read_bytes() == before
    assert not (settings_tmp.home / "models" / source.name).exists()
    del normal


def test_model_import_reconciles_interrupted_staging(settings_tmp: Any) -> None:
    service = _configure(settings_tmp)
    service._prepare_directories()
    staging = service.staging_dir / "restart-model.gguf.part"
    staging.write_bytes(b"partial")
    snapshot = service.journal_dir / "restart.models.yaml.before"
    snapshot.write_bytes(service.config_path.read_bytes())
    journal = {
        "schema_version": 1,
        "operation_id": "restart",
        "status": "copying",
        "staging_path": str(staging),
        "destination_path": str(service.models_dir / "restart.gguf"),
        "snapshot_path": str(snapshot),
        "sha256": None,
    }
    (service.journal_dir / "restart.json").write_text(
        json.dumps(journal),
        encoding="utf-8",
    )
    assert reconcile_model_imports(settings_tmp) == ["restart"]
    assert not staging.exists()
    assert not snapshot.exists()


@pytest.mark.asyncio
async def test_model_import_concurrency_and_retry_are_idempotent(settings_tmp: Any) -> None:
    service = _configure(settings_tmp)
    source = _gguf(settings_tmp.home / "incoming" / "same.gguf", size=64)
    outcomes = await asyncio.gather(
        _run(service, source, operation_id="concurrent-a", model_id="candidate-a"),
        _run(service, source, operation_id="concurrent-b", model_id="candidate-b"),
        return_exceptions=True,
    )
    assert sum(isinstance(item, dict) for item in outcomes) == 1
    assert any(
        isinstance(item, ModelImportError) and "overwrite_rejected" in str(item)
        for item in outcomes
    )
    succeeded = next(item for item in outcomes if isinstance(item, dict))
    operation_id = "concurrent-a" if succeeded["model_id"] == "candidate-a" else "concurrent-b"
    source.unlink()

    async def noop(_percent: int, _code: str) -> None:
        return None

    repeated = await service.run(
        operation_id=operation_id,
        source_path=str(source),
        model_id=str(succeeded["model_id"]),
        role="brain",
        name="Candidate",
        expected_sha256=str(succeeded["sha256"]),
        cancellation_event=asyncio.Event(),
        progress=noop,
    )
    assert repeated == succeeded


@pytest.mark.asyncio
async def test_model_import_never_opens_a_network_socket(
    settings_tmp: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _configure(settings_tmp)
    source = _gguf(settings_tmp.home / "incoming" / "offline.gguf", size=64)

    def deny_socket(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("model import must not create a network socket")

    monkeypatch.setattr(socket, "socket", deny_socket)
    result = await _run(
        service,
        source,
        operation_id="offline-import",
        model_id="offline-candidate",
    )
    assert result["registration_status"] == "registered_inactive"
