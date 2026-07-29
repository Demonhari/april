from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from april_common.path_security import deny_sensitive_path, is_path_within_roots
from april_common.settings import AprilSettings
from april_common.time import utc_now_iso
from services.april_runtime.model_registry import ModelRegistry

PLAN_FORMAT_VERSION = 1
MAX_DATASET_BYTES = 100 * 1024 * 1024
MAX_ROW_BYTES = 1024 * 1024
_REDACTED = "[REDACTED]"

_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:sk|ghp|github_pat|xox[baprs])[-_][A-Za-z0-9_-]{12,}\b"),
    re.compile(
        r"(?i)\b(?:api[_ -]?key|access[_ -]?token|token|password|secret)\b"
        r"\s*[:=]\s*[^\s,;]{4,}"
    ),
    re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?-----END .*?PRIVATE KEY-----",
        re.S,
    ),
    re.compile(
        r"(?i)(?:/Users/[^/\s]+|/home/[^/\s]+)/(?:\.ssh|\.aws|\.config/gcloud|Library/Keychains)"
        r"(?:/[^\s\"']*)?"
    ),
)


class _ChatRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["chat"]
    prompt: str = Field(min_length=1, max_length=500_000)
    response: str = Field(min_length=1, max_length=500_000)
    conversation_id: str | None = Field(default=None, max_length=128)


class _PreferenceRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["preference"]
    prompt: str = Field(min_length=1, max_length=500_000)
    chosen: str = Field(min_length=1, max_length=500_000)
    rejected: str = Field(min_length=1, max_length=500_000)
    conversation_id: str | None = Field(default=None, max_length=128)


class _MemoryRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["memory"]
    content: str = Field(min_length=1, max_length=500_000)
    kind: str = Field(min_length=1, max_length=64)
    confidence: float = Field(ge=0.0, le=1.0)
    memory_id: str | None = Field(default=None, max_length=128)


_ROW_ADAPTER: TypeAdapter[_ChatRow | _PreferenceRow | _MemoryRow] = TypeAdapter(
    _ChatRow | _PreferenceRow | _MemoryRow
)


@dataclass(frozen=True, slots=True)
class FinetunePlan:
    format_version: int
    plan_id: str
    created_at: str
    source_basename: str
    dataset_sha256: str
    train_sha256: str
    evaluation_sha256: str
    configuration_sha256: str
    base_model_id: str
    base_model_sha256: str
    trainer_sha256: str | None
    evaluator_sha256: str | None
    sample_count: int
    train_count: int
    evaluation_count: int
    redaction_count: int
    trainer_configured: bool
    evaluator_configured: bool
    adapter_candidate_basename: str
    status: str = "awaiting_approval"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_finetune_plan(
    settings: AprilSettings,
    *,
    source: Path,
    base_model_id: str,
) -> FinetunePlan:
    """Validate, redact and deterministically split a reviewed JSONL dataset."""
    source_path = source.expanduser().resolve(strict=True)
    deny_sensitive_path(source_path)
    if not source_path.is_file() or not stat.S_ISREG(source_path.stat().st_mode):
        raise ValueError("Fine-tune source must be a regular JSONL file.")
    if source_path.suffix.casefold() != ".jsonl":
        raise ValueError("Fine-tune source must use the .jsonl extension.")
    if source_path.stat().st_size > MAX_DATASET_BYTES:
        raise ValueError("Fine-tune source exceeds the configured safety bound.")
    if not is_path_within_roots(source_path, [settings.home, *settings.allowed_roots]):
        raise ValueError("Fine-tune source is outside APRIL's configured allowed roots.")

    registry = ModelRegistry.from_file(
        settings.home / "configs" / "models.yaml",
        root=settings.home,
    )
    model = registry.get(base_model_id)
    base_path = model.resolved_path(registry.root)
    if not base_path.is_file():
        raise ValueError("Configured fine-tune base model is unavailable.")

    rows, redactions = _load_and_redact_rows(source_path)
    if len(rows) < settings.finetune.minimum_samples:
        raise ValueError(
            f"Fine-tune dataset requires at least {settings.finetune.minimum_samples} samples."
        )
    configuration = _configuration_payload(settings, base_model_id)
    configuration_sha = _sha256_json(configuration)
    dataset_sha = _sha256_rows(rows)
    plan_id = hashlib.sha256(
        f"{dataset_sha}:{configuration_sha}:{_sha256_file(base_path)}".encode()
    ).hexdigest()[:32]
    train, evaluation = _deterministic_split(
        rows,
        fraction=settings.finetune.evaluation_fraction,
        seed=plan_id,
    )
    if not train or not evaluation:
        raise ValueError("Fine-tune split must contain separate train and evaluation samples.")

    plan_root = settings.evolution_path / "finetune" / "plans" / plan_id
    plan_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(plan_root, 0o700)
    train_path = plan_root / "train.jsonl"
    evaluation_path = plan_root / "evaluation.jsonl"
    _atomic_write(train_path, _rows_text(train))
    _atomic_write(evaluation_path, _rows_text(evaluation))

    trainer_hash = _optional_executable_hash(settings.finetune.trainer_executable)
    evaluator_hash = _optional_executable_hash(settings.finetune.evaluator_executable)
    candidate_basename = f"{base_model_id}-{plan_id}.gguf"
    plan = FinetunePlan(
        format_version=PLAN_FORMAT_VERSION,
        plan_id=plan_id,
        created_at=utc_now_iso(),
        source_basename=source_path.name,
        dataset_sha256=dataset_sha,
        train_sha256=_sha256_file(train_path),
        evaluation_sha256=_sha256_file(evaluation_path),
        configuration_sha256=configuration_sha,
        base_model_id=base_model_id,
        base_model_sha256=_sha256_file(base_path),
        trainer_sha256=trainer_hash,
        evaluator_sha256=evaluator_hash,
        sample_count=len(rows),
        train_count=len(train),
        evaluation_count=len(evaluation),
        redaction_count=redactions,
        trainer_configured=trainer_hash is not None,
        evaluator_configured=evaluator_hash is not None,
        adapter_candidate_basename=candidate_basename,
    )
    _atomic_write(
        plan_root / "manifest.json",
        json.dumps(plan.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
    )
    return plan


def load_finetune_plan(settings: AprilSettings, plan_id: str) -> FinetunePlan:
    if not re.fullmatch(r"[a-f0-9]{16,64}", plan_id):
        raise ValueError("Invalid fine-tune plan identifier.")
    path = settings.evolution_path / "finetune" / "plans" / plan_id / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        plan = FinetunePlan(**payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Fine-tune plan is missing or malformed.") from exc
    if plan.format_version != PLAN_FORMAT_VERSION or plan.plan_id != plan_id:
        raise ValueError("Fine-tune plan format or identifier is invalid.")
    return plan


def _load_and_redact_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    redaction_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if len(line.encode("utf-8")) > MAX_ROW_BYTES:
                raise ValueError(f"Fine-tune row {line_number} exceeds the safety bound.")
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                validated = _ROW_ADAPTER.validate_python(raw).model_dump(mode="json")
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"Fine-tune row {line_number} is malformed or unsupported."
                ) from exc
            redacted, count = _redact_value(validated)
            redaction_count += count
            rows.append(redacted)
    if not rows:
        raise ValueError("Fine-tune dataset is empty.")
    return rows, redaction_count


def _redact_value(value: Any) -> tuple[Any, int]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        count = 0
        for key, child in value.items():
            redacted, child_count = _redact_value(child)
            result[str(key)] = redacted
            count += child_count
        return result, count
    if isinstance(value, list):
        result_list: list[Any] = []
        count = 0
        for child in value:
            redacted, child_count = _redact_value(child)
            result_list.append(redacted)
            count += child_count
        return result_list, count
    if isinstance(value, str):
        text = value
        count = 0
        for pattern in _SECRET_PATTERNS:
            text, substitutions = pattern.subn(_REDACTED, text)
            count += substitutions
        return text, count
    return value, 0


def _deterministic_split(
    rows: list[dict[str, Any]], *, fraction: float, seed: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            (
                f"{seed}:"
                f"{json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=False)}"
            ).encode()
        ).hexdigest(),
    )
    evaluation_count = max(1, min(len(ordered) - 1, round(len(ordered) * fraction)))
    return ordered[evaluation_count:], ordered[:evaluation_count]


def _configuration_payload(settings: AprilSettings, model_id: str) -> dict[str, Any]:
    return {
        "base_model_id": model_id,
        "minimum_samples": settings.finetune.minimum_samples,
        "evaluation_fraction": settings.finetune.evaluation_fraction,
        "trainer_basename": (
            settings.finetune.trainer_executable.name
            if settings.finetune.trainer_executable is not None
            else None
        ),
        "trainer_arguments": settings.finetune.trainer_arguments,
        "evaluator_basename": (
            settings.finetune.evaluator_executable.name
            if settings.finetune.evaluator_executable is not None
            else None
        ),
        "evaluator_arguments": settings.finetune.evaluator_arguments,
    }


def _optional_executable_hash(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve(strict=True)
    deny_sensitive_path(resolved)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError("Configured fine-tune executable is not an executable regular file.")
    return _sha256_file(resolved)


def _rows_text(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )


def _sha256_rows(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_rows_text(rows).encode("utf-8")).hexdigest()


def _sha256_json(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)
