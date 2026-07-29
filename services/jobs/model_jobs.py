from __future__ import annotations

import asyncio
import hashlib
import json
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from april_common.path_security import is_path_within_roots
from april_common.process_environment import ProcessCategory
from april_common.process_runner import (
    ProcessStatus,
    ResourceLimitProfile,
    run_restricted_process,
)
from april_common.settings import AprilSettings
from services.april_runtime.model_registry import ModelRegistry

MIN_GGUF_BYTES = 4
MAX_SAFE_MODEL_RESULT_BYTES = 64 * 1024


class ModelJobError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ValidatedRegisteredModel:
    model_id: str
    role: str
    path: Path
    basename: str
    size: int
    sha256: str


def validate_registered_model(
    settings: AprilSettings,
    model_id: str,
    *,
    cancellation_event: asyncio.Event | None = None,
) -> ValidatedRegisteredModel:
    registry = ModelRegistry.from_file(
        settings.home / "configs" / "models.yaml",
        root=settings.home,
    )
    model = registry.get(model_id)
    path = model.resolved_path(registry.root)
    if not is_path_within_roots(path, [settings.home, *settings.allowed_roots]):
        raise ModelJobError("registered_model_outside_allowed_roots")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise ModelJobError("registered_model_unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ModelJobError("registered_model_not_regular_file")
    if metadata.st_size < MIN_GGUF_BYTES:
        raise ModelJobError("registered_model_too_small")
    try:
        with path.open("rb") as handle:
            if handle.read(4) != b"GGUF":
                raise ModelJobError("registered_model_invalid_gguf_magic")
            digest = hashlib.sha256()
            handle.seek(0)
            while chunk := handle.read(1024 * 1024):
                if cancellation_event is not None and cancellation_event.is_set():
                    raise asyncio.CancelledError
                digest.update(chunk)
    except OSError as exc:
        raise ModelJobError("registered_model_unreadable") from exc
    return ValidatedRegisteredModel(
        model_id=model.id,
        role=str(model.role),
        path=path,
        basename=path.name,
        size=metadata.st_size,
        sha256=digest.hexdigest(),
    )


async def run_model_utility_job(
    settings: AprilSettings,
    *,
    model_id: str,
    mode: str,
    cancellation_event: asyncio.Event,
    timeout_seconds: float,
) -> dict[str, Any]:
    if mode not in {"verify", "benchmark"}:
        raise ValueError("unknown_model_utility_mode")
    validated = validate_registered_model(
        settings,
        model_id,
        cancellation_event=cancellation_event,
    )
    result = await run_restricted_process(
        [
            sys.executable,
            "-m",
            "apps.runner.model_job_worker",
            "--home",
            str(settings.home),
            "--model-id",
            model_id,
            "--mode",
            mode,
        ],
        cwd=settings.home,
        category=(
            ProcessCategory.MODEL_VERIFICATION if mode == "verify" else ProcessCategory.BENCHMARKING
        ),
        timeout_seconds=timeout_seconds,
        max_stdout_bytes=MAX_SAFE_MODEL_RESULT_BYTES,
        max_stderr_bytes=8_192,
        cancellation_event=cancellation_event,
        resource_limit_profile=ResourceLimitProfile.MODEL_UTILITY,
        april_home=settings.home,
    )
    if result.status is ProcessStatus.CANCELLED:
        raise asyncio.CancelledError
    if result.status is ProcessStatus.TIMED_OUT:
        raise ModelJobError(f"model_{mode}_timeout")
    if result.status is not ProcessStatus.COMPLETED or result.returncode != 0:
        raise ModelJobError(f"model_{mode}_failed")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ModelJobError(f"model_{mode}_result_invalid") from exc
    if not isinstance(payload, dict) or payload.get("model_id") != model_id:
        raise ModelJobError(f"model_{mode}_result_invalid")
    return {
        **payload,
        "model_id": validated.model_id,
        "role": validated.role,
        "model_basename": validated.basename,
        "model_size": validated.size,
        "model_sha256": validated.sha256,
        "stdout_truncated": result.stdout_truncated,
        "stderr_truncated": result.stderr_truncated,
        "resource_limits_applied": list(result.resource_limits.applied),
    }
