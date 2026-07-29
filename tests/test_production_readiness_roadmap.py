from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from apps.runner.release_tools import (
    ReleaseValidationError,
    build_production_app,
    run_apple_tool,
    validate_app_bundle,
    validate_release_zip,
)
from apps.runner.speaker_live import (
    enable_soft_speaker_gate,
    run_speaker_live_verification,
    write_speaker_live_report,
)
from april_common.credentials import InMemoryCredentialStore
from april_common.process_runner import (
    ProcessStatus,
    ResourceLimitProfile,
    ResourceLimitReport,
    RestrictedProcessResult,
)
from services.api.server import create_app
from services.finetune.dataset import create_finetune_plan
from services.jobs.finetune_job import run_finetune_job
from services.jobs.model_jobs import ModelJobError, validate_registered_model
from services.jobs.registry import default_job_registry
from services.jobs.schemas import JobStatus
from services.jobs.store import JobStore, JobTransitionError
from services.memory.database import Database
from services.memory.encryption import (
    MemoryEncryptionError,
    SensitiveMemoryEncryption,
    provision_memory_key,
    rotate_memory_key,
)
from services.memory.migrations import run_migrations
from services.memory.sqlite_memory import SqliteMemory
from services.memory.writer import MemoryWriter
from services.voice.microphone import Microphone, write_pcm_wav
from services.wake.speaker import SpeakerVerifier


class FakeAuthenticatedEncryption:
    def encrypt(self, key: bytes, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
        tag = hashlib.sha256(key + nonce + aad + plaintext).digest()
        return tag + plaintext

    def decrypt(self, key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
        plaintext = ciphertext[32:]
        expected = hashlib.sha256(key + nonce + aad + plaintext).digest()
        if ciphertext[:32] != expected:
            raise MemoryEncryptionError("authentication failed")
        return plaintext


class FakeMicrophone(Microphone):
    async def record_push_to_talk(self, output_path: Path) -> Path:
        return write_pcm_wav(output_path, [b"\x01\x00"] * 100)


class FakeSpeakerVerifier(SpeakerVerifier):
    def score(self, enrollment_paths: list[Path], pcm: bytes) -> float:
        del enrollment_paths
        return 0.9 if pcm.startswith(b"\x01\x00") else 0.1


def _configured_home(settings_tmp, *, minimum_samples: int = 4) -> object:
    home = settings_tmp.home
    (home / "configs").mkdir(exist_ok=True)
    shutil.copy(Path.cwd() / "configs" / "april.yaml", home / "configs" / "april.yaml")
    model_path = home / "models" / "base.gguf"
    model_path.parent.mkdir()
    model_path.write_bytes(b"GGUF" + b"\x00" * 128)
    models = {
        "models": {
            "base": {
                "id": "base",
                "name": "Local Base",
                "path": str(model_path),
                "backend": "llama_cpp",
                "role": "brain",
                "threads": 2,
                "context_size": 1024,
                "temperature": 0.1,
                "max_output_tokens": 32,
            }
        }
    }
    (home / "configs" / "models.yaml").write_text(
        yaml.safe_dump(models),
        encoding="utf-8",
    )
    trainer = home / "trainer"
    evaluator = home / "evaluator"
    trainer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    evaluator.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    trainer.chmod(0o700)
    evaluator.chmod(0o700)
    return settings_tmp.model_copy(
        update={
            "finetune": settings_tmp.finetune.model_copy(
                update={
                    "enabled": True,
                    "minimum_samples": minimum_samples,
                    "trainer_executable": trainer,
                    "trainer_arguments": ["{output_adapter}"],
                    "evaluator_executable": evaluator,
                    "evaluator_arguments": ["{candidate_adapter}"],
                }
            )
        }
    )


def test_new_job_types_are_gated_and_have_safe_recovery_policy() -> None:
    disabled = default_job_registry()
    assert disabled.require("model_import_verification").restart_safe is True
    assert disabled.require("model_benchmark").cancellation_behavior == "process_group"
    with pytest.raises(ValueError, match="finetune_job_disabled"):
        disabled.require("finetune")
    with pytest.raises(ValueError, match="dream_cycle_job_unavailable"):
        disabled.require("dream_cycle")

    enabled = default_job_registry(finetune_enabled=True, evolution_enabled=True)
    finetune = enabled.require("finetune")
    assert finetune.approval_required is True
    assert finetune.permission_level == 4
    assert finetune.restart_safe is False
    dream = enabled.require("dream_cycle")
    assert dream.maximum_attempts == 1
    assert dream.idempotent is False


@pytest.mark.asyncio
async def test_new_job_types_claim_once_cancel_recover_and_retry(settings_tmp) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    store = JobStore(
        database,
        default_job_registry(finetune_enabled=True, evolution_enabled=True),
    )
    payloads = {
        "model_import_verification": {"model_id": "base"},
        "model_benchmark": {"model_id": "base"},
        "finetune": {"plan_id": "a" * 32},
        "dream_cycle": {},
    }
    with pytest.raises(JobTransitionError, match="approval_required"):
        await store.submit(
            job_type="finetune",
            payload=payloads["finetune"],
            owner="local-user",
        )
    for job_type, payload in payloads.items():
        await store.submit(
            job_type=job_type,
            payload=payload,
            owner="local-user",
            approved=job_type == "finetune",
        )
    claims = await asyncio.gather(
        *(store.claim_next(worker_id=f"worker-{index}", lease_seconds=30) for index in range(8))
    )
    claimed = [claim for claim in claims if claim is not None]
    assert len(claimed) == 4
    assert len({claim.id for claim in claimed}) == 4

    by_type = {claim.job_type: claim for claim in claimed}
    finetune = by_type["finetune"]
    await database.execute(
        "UPDATE background_jobs SET lease_expires_at = '2000-01-01T00:00:00Z' WHERE id = ?",
        (finetune.id,),
    )
    recovered = await store.recover_expired_leases()
    recovered_finetune = next(item for item in recovered if item.id == finetune.id)
    assert recovered_finetune.status is JobStatus.INTERRUPTED
    retried, already_queued = await store.retry(finetune.id)
    assert retried.status is JobStatus.QUEUED
    assert already_queued is False

    dream = by_type["dream_cycle"]
    await store.finish(
        dream.id,
        worker_id=str(dream.worker_id),
        status=JobStatus.FAILED,
        error_code="review_gate_failed",
    )
    with pytest.raises(JobTransitionError, match="retry_not_eligible"):
        await store.retry(dream.id)

    queued_cancel = await store.submit(
        job_type="model_benchmark",
        payload={"model_id": "cancel-me"},
        owner="local-user",
    )
    cancelled, already_terminal = await store.request_cancel(queued_cancel.id)
    assert cancelled.status is JobStatus.CANCELLED
    assert already_terminal is False


def test_registered_model_validation_rejects_non_gguf(settings_tmp) -> None:
    settings = _configured_home(settings_tmp)
    model_path = settings.home / "models" / "base.gguf"
    model_path.write_bytes(b"NOPE")
    with pytest.raises(ModelJobError, match="registered_model_invalid_gguf_magic"):
        validate_registered_model(settings, "base")


def test_finetune_plan_redacts_and_splits_deterministically(settings_tmp) -> None:
    settings = _configured_home(settings_tmp)
    source = settings.home / "reviewed.jsonl"
    rows = [
        {
            "type": "chat",
            "prompt": f"prompt {index} token=super-secret-value",
            "response": f"response {index} /Users/operator/.ssh/id_ed25519",
        }
        for index in range(6)
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    first = create_finetune_plan(settings, source=source, base_model_id="base")
    second = create_finetune_plan(settings, source=source, base_model_id="base")
    assert first.train_sha256 == second.train_sha256
    assert first.evaluation_sha256 == second.evaluation_sha256
    assert first.train_count > 0
    assert first.evaluation_count > 0
    assert first.redaction_count >= 12
    plan_dir = settings.home / "data" / "evolution" / "finetune" / "plans"
    combined = "\n".join(path.read_text(encoding="utf-8") for path in plan_dir.rglob("*.jsonl"))
    assert "super-secret-value" not in combined
    assert "/Users/operator/.ssh" not in combined


@pytest.mark.asyncio
async def test_finetune_job_registers_candidate_without_activation(
    settings_tmp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _configured_home(settings_tmp)
    source = settings.home / "reviewed.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "type": "chat",
                    "prompt": f"prompt {index}",
                    "response": f"response {index}",
                }
            )
            + "\n"
            for index in range(6)
        ),
        encoding="utf-8",
    )
    plan = create_finetune_plan(settings, source=source, base_model_id="base")

    async def fake_process(argv: list[str], **_kwargs: object) -> RestrictedProcessResult:
        if Path(argv[0]).name == "trainer":
            Path(argv[1]).write_bytes(b"GGUF" + b"\x00" * 64)
            stdout = ""
        else:
            perplexity = 10.0 if argv[1] == "__BASE__" else 9.0
            stdout = json.dumps({"perplexity": perplexity})
        return RestrictedProcessResult(
            status=ProcessStatus.COMPLETED,
            returncode=0,
            stdout=stdout,
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
            duration_seconds=0.01,
            failure_code=None,
            resource_limits=ResourceLimitReport(
                requested_profile=ResourceLimitProfile.TRAINING,
                applied=(),
                unsupported=(),
            ),
        )

    monkeypatch.setattr("services.jobs.finetune_job.run_restricted_process", fake_process)

    async def progress(_percent: int, _code: str) -> None:
        return None

    result = await run_finetune_job(
        settings,
        plan_id=plan.plan_id,
        cancellation_event=asyncio.Event(),
        progress=progress,
    )
    assert result["adapter_active"] is False
    manifest = json.loads(
        (settings.evolution_path / "adapters" / "candidates" / f"{plan.plan_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "inactive_candidate"
    assert not (settings.evolution_path / "adapters" / "active" / "base.json").exists()


@pytest.mark.asyncio
async def test_sensitive_memory_encryption_rotation_and_missing_key(settings_tmp) -> None:
    store = InMemoryCredentialStore()
    provision_memory_key(store)
    algorithm = FakeAuthenticatedEncryption()
    from services.memory.encryption import _load_keyring

    cipher = SensitiveMemoryEncryption(_load_keyring(store), encryption=algorithm)
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(
        database,
        sensitive_encryption=cipher,
        sensitive_encryption_enabled=True,
    )
    writer = MemoryWriter(memory, sensitive_encryption_available=True)
    record = await writer.write(
        "password=local-only",
        reason="password label local-only",
        requested_by_user=True,
        sensitive=True,
    )
    raw = await database.fetchone(
        "SELECT content, reason FROM memories WHERE id = ?",
        (record.id,),
    )
    assert raw is not None
    assert "local-only" not in str(raw["content"])
    assert "local-only" not in str(raw["reason"])
    assert (await memory.get_memory(record.id)).content == "password=local-only"  # type: ignore[union-attr]

    settings = settings_tmp.model_copy(
        update={
            "memory": settings_tmp.memory.model_copy(update={"sensitive_encryption_enabled": True})
        }
    )
    result = await rotate_memory_key(settings, database, store=store, encryption=algorithm)
    assert result["rotated_records"] == 1
    rotated_cipher = SensitiveMemoryEncryption(_load_keyring(store), encryption=algorithm)
    rotated_memory = SqliteMemory(database, sensitive_encryption=rotated_cipher)
    assert (await rotated_memory.get_memory(record.id)).content == "password=local-only"  # type: ignore[union-attr]

    unavailable = SqliteMemory(database)
    unavailable_record = await unavailable.get_memory(record.id)
    assert unavailable_record is not None
    assert "unavailable" in unavailable_record.content

    encrypted_row = await database.fetchone(
        "SELECT content FROM memories WHERE id = ?",
        (record.id,),
    )
    assert encrypted_row is not None
    encrypted_value = str(encrypted_row["content"])
    corrupted = encrypted_value[:-1] + ("A" if encrypted_value[-1] != "A" else "B")
    await database.execute(
        "UPDATE memories SET content = ? WHERE id = ?",
        (corrupted, record.id),
    )
    with pytest.raises(MemoryEncryptionError):
        await rotated_memory.get_memory(record.id)


def test_release_zip_rejects_models_credentials_and_generated_data(tmp_path: Path) -> None:
    safe = tmp_path / "safe.zip"
    with zipfile.ZipFile(safe, "w") as archive:
        archive.writestr("APRIL/README.md", "safe")
    assert validate_release_zip(safe) == ("APRIL/README.md",)

    for member in ("models/model.gguf", ".venv/bin/python", "credentials.json"):
        bad = tmp_path / f"{member.replace('/', '-')}.zip"
        with zipfile.ZipFile(bad, "w") as archive:
            archive.writestr(member, "not safe")
        with pytest.raises(ReleaseValidationError):
            validate_release_zip(bad)


def test_production_bundle_and_apple_tool_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_production_app(tmp_path / "APRIL.app", version="1.2.3")
    validate_app_bundle(app)
    assert stat.S_IMODE((app / "Contents" / "MacOS" / "APRIL").stat().st_mode) == 0o755
    combined = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in app.rglob("*")
        if path.is_file()
    )
    assert "APRIL_API_TOKEN" not in combined

    captured: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        captured.update({"argv": argv, **kwargs})
        return SimpleNamespace(returncode=0, stdout="token=hidden", stderr="")

    monkeypatch.setattr("apps.runner.release_tools.subprocess.run", fake_run)
    result = run_apple_tool(["/usr/bin/codesign", "--verify", str(app)])
    assert captured["shell"] is False
    assert set(captured["env"]) <= {"PATH", "LANG", "LC_ALL", "HOME"}  # type: ignore[arg-type]
    assert "hidden" not in result.output


@pytest.mark.asyncio
async def test_speaker_live_fake_backend_is_numeric_and_discards_audio(settings_tmp) -> None:
    home = settings_tmp.home
    shutil.copytree(Path.cwd() / "configs", home / "configs")
    model = home / "speaker.onnx"
    model.write_bytes(b"fake")
    profiles = home / "data" / "voice_profiles"
    rejected = profiles / "rejected"
    rejected.mkdir(parents=True)
    write_pcm_wav(profiles / "accepted.wav", [b"\x01\x00"] * 100)
    write_pcm_wav(rejected / "rejected.wav", [b"\x02\x00"] * 100)
    settings = settings_tmp.model_copy(
        update={"wake": settings_tmp.wake.model_copy(update={"speaker_verifier_model_path": model})}
    )
    report = await run_speaker_live_verification(
        settings=settings,
        confirm_capture=lambda _: True,
        microphone=FakeMicrophone(),
        verifier=FakeSpeakerVerifier(),
    )
    assert report.speaker_live_verified is True
    assert report.false_accept_fixture_passed is True
    assert report.false_reject_fixture_passed is True
    assert not list(settings.audio_cache_path.glob("speaker-live-*.wav"))
    report.generated_at = "2000-01-01T00:00:00Z"
    stale_path = home / "data" / "verification" / "speaker-live.json"
    write_speaker_live_report(report, stale_path)
    with pytest.raises(ValueError, match="fresh successful"):
        enable_soft_speaker_gate(settings, stale_path)


def test_route_modules_preserve_public_paths() -> None:
    paths = {route.path for route in create_app().routes}
    assert {
        "/health",
        "/chat",
        "/chat/stream",
        "/jobs",
        "/jobs/{job_id}",
        "/voice/input",
        "/wake",
        "/sessions",
        "/agents/run",
        "/tools/request",
        "/tools/approve",
        "/memory",
        "/memory/search",
    } <= paths
