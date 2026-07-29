from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from april_common.process_environment import ProcessCategory
from april_common.process_runner import (
    ProcessStatus,
    ResourceLimitProfile,
    run_restricted_process,
)
from april_common.settings import AprilSettings
from april_common.time import utc_now_iso
from services.april_runtime.model_registry import ModelRegistry
from services.finetune.dataset import FinetunePlan, load_finetune_plan

Progress = Callable[[int, str], Awaitable[None]]


class FinetuneJobError(RuntimeError):
    pass


async def run_finetune_job(
    settings: AprilSettings,
    *,
    plan_id: str,
    cancellation_event: asyncio.Event,
    progress: Progress,
) -> dict[str, Any]:
    if not settings.finetune.enabled:
        raise FinetuneJobError("finetune_disabled")
    plan = load_finetune_plan(settings, plan_id)
    paths = _validated_plan_paths(settings, plan)
    trainer = _validated_executable(
        settings.finetune.trainer_executable,
        expected_sha256=plan.trainer_sha256,
        label="trainer",
    )
    evaluator = _validated_executable(
        settings.finetune.evaluator_executable,
        expected_sha256=plan.evaluator_sha256,
        label="evaluator",
    )
    await progress(10, "finetune_validated")
    trainer_argv = [
        str(trainer),
        *_render_arguments(
            settings.finetune.trainer_arguments,
            {
                "base_model": paths["base_model"],
                "train_dataset": paths["train"],
                "eval_dataset": paths["evaluation"],
                "output_adapter": paths["candidate"],
                "candidate_adapter": paths["candidate"],
            },
        ),
    ]
    training = await run_restricted_process(
        trainer_argv,
        cwd=settings.home,
        category=ProcessCategory.FINETUNE,
        timeout_seconds=settings.finetune.timeout_seconds,
        max_stdout_bytes=settings.finetune.max_output_bytes,
        max_stderr_bytes=settings.finetune.max_output_bytes,
        cancellation_event=cancellation_event,
        resource_limit_profile=ResourceLimitProfile.TRAINING,
        april_home=settings.home,
    )
    if training.status is ProcessStatus.CANCELLED:
        raise asyncio.CancelledError
    if training.status is ProcessStatus.TIMED_OUT:
        raise FinetuneJobError("trainer_timeout")
    if training.status is not ProcessStatus.COMPLETED or training.returncode != 0:
        raise FinetuneJobError("trainer_failed")
    candidate = Path(paths["candidate"])
    try:
        with candidate.open("rb") as handle:
            magic = handle.read(4)
    except OSError:
        magic = b""
    if not candidate.is_file() or magic != b"GGUF":
        raise FinetuneJobError("candidate_adapter_missing_or_invalid")
    os.chmod(candidate, 0o600)
    await progress(65, "finetune_training_completed")

    base_perplexity = await _evaluate(
        settings,
        evaluator=evaluator,
        plan_paths=paths,
        target="base",
        cancellation_event=cancellation_event,
    )
    await progress(80, "finetune_baseline_evaluated")
    candidate_perplexity = await _evaluate(
        settings,
        evaluator=evaluator,
        plan_paths=paths,
        target="candidate",
        cancellation_event=cancellation_event,
    )
    await progress(90, "finetune_candidate_evaluated")

    adapter_sha = _sha256_file(candidate)
    evidence = {
        "schema_version": 1,
        "evidence_type": "lora_perplexity",
        "plan_id": plan.plan_id,
        "model_id": plan.base_model_id,
        "adapter_basename": candidate.name,
        "adapter_sha256": adapter_sha,
        "base_perplexity": base_perplexity,
        "adapter_perplexity": candidate_perplexity,
        "heldout_dataset_sha256": plan.evaluation_sha256,
        "dataset_sha256": plan.dataset_sha256,
        "configuration_sha256": plan.configuration_sha256,
        "base_model_sha256": plan.base_model_sha256,
        "trainer_sha256": plan.trainer_sha256,
        "evaluator_sha256": plan.evaluator_sha256,
        "created_at": utc_now_iso(),
        "activation_eligible_from_metrics": candidate_perplexity <= base_perplexity,
        "human_review_required": True,
        "active": False,
    }
    evidence_path = settings.evolution_path / "adapters" / "evidence" / f"{plan.plan_id}.json"
    candidate_manifest = (
        settings.evolution_path / "adapters" / "candidates" / f"{plan.plan_id}.json"
    )
    _atomic_json(evidence_path, evidence)
    _atomic_json(
        candidate_manifest,
        {
            "schema_version": 1,
            "candidate_id": plan.plan_id,
            "model_id": plan.base_model_id,
            "adapter_basename": candidate.name,
            "adapter_sha256": adapter_sha,
            "evidence_basename": evidence_path.name,
            "status": "inactive_candidate",
            "created_at": utc_now_iso(),
        },
    )
    await progress(100, "finetune_candidate_registered_inactive")
    return {
        "plan_id": plan.plan_id,
        "model_id": plan.base_model_id,
        "adapter_basename": candidate.name,
        "adapter_sha256": adapter_sha,
        "evidence_basename": evidence_path.name,
        "candidate_manifest_basename": candidate_manifest.name,
        "base_perplexity": base_perplexity,
        "candidate_perplexity": candidate_perplexity,
        "activation_eligible_from_metrics": candidate_perplexity <= base_perplexity,
        "adapter_active": False,
    }


def _validated_plan_paths(settings: AprilSettings, plan: FinetunePlan) -> dict[str, str]:
    root = settings.evolution_path / "finetune" / "plans" / plan.plan_id
    train = root / "train.jsonl"
    evaluation = root / "evaluation.jsonl"
    if _sha256_file(train) != plan.train_sha256:
        raise FinetuneJobError("training_dataset_hash_mismatch")
    if _sha256_file(evaluation) != plan.evaluation_sha256:
        raise FinetuneJobError("evaluation_dataset_hash_mismatch")
    registry = ModelRegistry.from_file(
        settings.home / "configs" / "models.yaml",
        root=settings.home,
    )
    base_model = registry.get(plan.base_model_id).resolved_path(registry.root)
    if _sha256_file(base_model) != plan.base_model_sha256:
        raise FinetuneJobError("base_model_hash_mismatch")
    candidate_root = settings.evolution_path / "adapters" / "candidates"
    candidate_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    candidate = candidate_root / plan.adapter_candidate_basename
    return {
        "base_model": str(base_model),
        "train": str(train),
        "evaluation": str(evaluation),
        "candidate": str(candidate),
    }


async def _evaluate(
    settings: AprilSettings,
    *,
    evaluator: Path,
    plan_paths: Mapping[str, str],
    target: str,
    cancellation_event: asyncio.Event,
) -> float:
    values = {
        "base_model": plan_paths["base_model"],
        "train_dataset": plan_paths["train"],
        "eval_dataset": plan_paths["evaluation"],
        "output_adapter": plan_paths["candidate"],
        "candidate_adapter": plan_paths["candidate"] if target == "candidate" else "__BASE__",
    }
    result = await run_restricted_process(
        [str(evaluator), *_render_arguments(settings.finetune.evaluator_arguments, values)],
        cwd=settings.home,
        category=ProcessCategory.FINETUNE,
        timeout_seconds=min(settings.finetune.timeout_seconds, 14_400.0),
        max_stdout_bytes=settings.finetune.max_output_bytes,
        max_stderr_bytes=settings.finetune.max_output_bytes,
        cancellation_event=cancellation_event,
        resource_limit_profile=ResourceLimitProfile.MODEL_UTILITY,
        april_home=settings.home,
    )
    if result.status is ProcessStatus.CANCELLED:
        raise asyncio.CancelledError
    if result.status is not ProcessStatus.COMPLETED or result.returncode != 0:
        raise FinetuneJobError(f"{target}_evaluator_failed")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1])
        value = float(payload["perplexity"])
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FinetuneJobError(f"{target}_perplexity_missing") from exc
    if not math.isfinite(value) or value <= 0:
        raise FinetuneJobError(f"{target}_perplexity_invalid")
    return value


def _validated_executable(value: Path | None, *, expected_sha256: str | None, label: str) -> Path:
    if value is None or expected_sha256 is None:
        raise FinetuneJobError(f"{label}_not_configured")
    path = value.expanduser().resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FinetuneJobError(f"{label}_not_executable")
    if _sha256_file(path) != expected_sha256:
        raise FinetuneJobError(f"{label}_hash_mismatch")
    return path


def _render_arguments(template: list[str], values: Mapping[str, str]) -> list[str]:
    rendered: list[str] = []
    for argument in template:
        value = argument
        for key, replacement in values.items():
            value = value.replace(f"{{{key}}}", replacement)
        if "\x00" in value:
            raise FinetuneJobError("invalid_rendered_argument")
        rendered.append(value)
    return rendered


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FinetuneJobError("required_finetune_artifact_unavailable") from exc
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)
