from __future__ import annotations

import asyncio
from typing import Any, TypeVar

import typer
from rich.table import Table

from apps.cli.render import console
from apps.runner.commands import registry as _registry
from apps.runner.commands.composition import composition as _composition_api
from apps.runner.wake_live import run_sentinel_live_verification
from april_common.errors import ConfigError
from april_common.settings import load_settings
from april_common.text_normalization import (
    HASHED_TOKEN_IMPLEMENTATION_VERSION,
    LEXICAL_TOKENIZER_VERSION,
)
from services.april_runtime.client import RuntimeClient
from services.april_runtime.model_registry import ModelRegistry
from services.memory.embeddings import HashedTokenEmbedding

_T = TypeVar("_T")

run_wake_word_live_verification = run_sentinel_live_verification


@_registry.project_app.command("add")
def project_add(
    ctx: typer.Context,
    path: str,
    name: str | None = typer.Option(None, "--name"),
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    args = ["project", "add", path]
    if name:
        args.extend(["--name", name])
    _composition_api._delegate(
        args,
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.project_app.command("index")
def project_index(
    ctx: typer.Context,
    project_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["project", "index", project_id],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.memory_app.command("list")
def memory_list(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["memory", "list"],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.memory_app.command("search")
def memory_search(
    ctx: typer.Context,
    query: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["memory", "search", query],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.memory_app.command("delete")
def memory_delete(
    ctx: typer.Context,
    memory_id: str,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["memory", "delete", memory_id],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.memory_app.command("export")
def memory_export(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    _composition_api._delegate(
        ["memory", "export"],
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.memory_app.command("reindex")
def memory_reindex(
    ctx: typer.Context,
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
    wait: bool = typer.Option(False, "--wait"),
    json_output: bool = typer.Option(False, "--json"),
    wait_timeout: float = typer.Option(3600.0, "--wait-timeout", min=1.0, max=86400.0),
) -> None:
    args = ["memory", "reindex"]
    if wait:
        args.append("--wait")
    if json_output:
        args.append("--json")
    if wait_timeout != 3600.0:
        args.extend(["--wait-timeout", str(wait_timeout)])
    _composition_api._delegate(
        args,
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.memory_app.command("repair-index")
def memory_repair_index(
    ctx: typer.Context,
    apply: bool = typer.Option(False, "--apply", help="Apply the reported pointer repair."),
    fake: bool = typer.Option(False, "--fake", help="Start missing services with fake runtime."),
) -> None:
    args = ["memory", "repair-index"]
    if apply:
        args.append("--apply")
    _composition_api._delegate(
        args,
        fake=_composition_api._effective_fake(ctx, fake),
        oneshot=_composition_api._effective_oneshot(ctx),
    )


@_registry.memory_app.command("doctor")
def memory_doctor(
    json_output: bool = typer.Option(False, "--json"),
    verify_runtime_embedding: bool = typer.Option(
        False,
        "--verify-runtime-embedding",
        help="Explicitly call April Runtime /runtime/embed to verify local semantic embeddings.",
    ),
) -> None:
    settings = load_settings(root=_composition_api._manager().home)
    data = _memory_doctor_report(settings, verify_runtime_embedding=verify_runtime_embedding)
    if json_output:
        console.print_json(data=data)
        return
    _print_memory_doctor(data)


def _memory_doctor_report(
    settings: Any, *, verify_runtime_embedding: bool = False
) -> dict[str, Any]:
    configured_provider = settings.memory.embedding_provider
    runtime_local_requested = configured_provider == "runtime-local"
    model_info = _embedding_model_info(settings)
    index = _vector_index_report(settings)
    verification: dict[str, Any] | None = None
    verified_dimensions: int | None = None
    verified_ok = False
    if verify_runtime_embedding and runtime_local_requested:
        verification = _composition_api._verify_runtime_embedding(
            settings, model_info.get("model_id")
        )
        verified_ok = verification.get("status") == "ok"
        raw_dimensions = verification.get("dimensions")
        verified_dimensions = raw_dimensions if type(raw_dimensions) is int else None

    model_ready = bool(
        model_info["embedding_model_registered"] and model_info["embedding_model_path_exists"]
    )
    fallback_to_hashed = runtime_local_requested and (
        not model_ready or (verification is not None and not verified_ok)
    )
    active_provider = "hashed-token" if fallback_to_hashed else configured_provider
    if active_provider == "hashed-token":
        active_dimensions: int | None = HashedTokenEmbedding().dimensions
    else:
        active_dimensions = verified_dimensions or index.get("persisted_dimensions")

    persisted_provider = index.get("persisted_provider")
    persisted_dimensions = index.get("persisted_dimensions")
    reindex_required = False
    if (persisted_provider is not None and persisted_provider != active_provider) or (
        persisted_dimensions is not None
        and active_dimensions is not None
        and persisted_dimensions != active_dimensions
    ):
        reindex_required = True

    index_status = str(index.get("status", "ok"))
    if reindex_required:
        status = "reindex_required"
    elif index_status == "not_ready":
        status = "not_ready"
    elif index_status == "degraded":
        status = "degraded"
    elif (runtime_local_requested and not model_ready) or (
        verification is not None and not verified_ok
    ):
        status = "not_ready"
    elif runtime_local_requested and verify_runtime_embedding and verified_ok:
        status = "ok"
    elif runtime_local_requested:
        status = "configured_unverified"
    else:
        status = "ok"

    warnings: list[str] = []
    if (
        configured_provider != "runtime-local"
        and getattr(settings, "environment", "development") == "production"
    ):
        warnings.append("runtime-local embeddings are not configured in production-like mode")
    if fallback_to_hashed:
        warnings.append("runtime-local embeddings fell back to hashed-token")
    if not model_info["embedding_model_registered"]:
        warnings.append("no embedding-role model is registered")

    report: dict[str, Any] = {
        "status": status,
        "configured_embedding_provider": configured_provider,
        "lexical_tokenizer_version": LEXICAL_TOKENIZER_VERSION,
        "hashed_token_implementation_version": HASHED_TOKEN_IMPLEMENTATION_VERSION,
        "hybrid_retrieval_enabled": True,
        "runtime_batch_embedding_capability": True,
        "runtime_batch_embedding_max_items": 64,
        "active_vector_index_provider": active_provider,
        "dimensions": active_dimensions,
        "runtime_local_requested": runtime_local_requested,
        "fell_back_to_hashed_token": fallback_to_hashed,
        "fallback_risk": runtime_local_requested and not verified_ok,
        "reindex_required": reindex_required,
        # The exact command to rebuild the index under the active provider. Always
        # present so switching providers never leaves the operator guessing; it is
        # the required next step whenever reindex_required is true.
        "reindex_command": "run april memory reindex",
        "repair_command": "run april memory repair-index --apply",
        "active_generation": index.get("effective_generation"),
        "last_successful_reindex_at": index.get("last_successful_reindex_at"),
        "embedding_model_id": model_info.get("model_id"),
        "embedding_role_model_registered": model_info["embedding_model_registered"],
        "embedding_model_path_exists": model_info["embedding_model_path_exists"],
        "embedding_model_path_basename": model_info.get("path_basename"),
        "vector_index": index,
        "warnings": warnings,
    }
    if runtime_local_requested and not (
        model_info["embedding_model_registered"] and model_info["embedding_model_path_exists"]
    ):
        # Highly visible setup hint when runtime-local is requested but unusable.
        report["setup_command"] = (
            "run april model import --role embedding --id april-embedding "
            "--name LOCAL_EMBEDDING --path /absolute/path/to/embedding.gguf "
            "--sha256 EXPECTED_SHA256"
        )
    if verification is not None:
        report["runtime_embedding_verification"] = verification
    return report


def _embedding_model_info(settings: Any) -> dict[str, Any]:
    try:
        registry = ModelRegistry.from_file(
            settings.home / "configs" / "models.yaml", root=settings.home
        )
    except ConfigError as exc:
        return {
            "embedding_model_registered": False,
            "embedding_model_path_exists": False,
            "model_id": settings.memory.embedding_model_id,
            "path_basename": None,
            "registry_error": str(exc),
        }
    candidates = [model for model in registry.list() if model.role == "embedding"]
    selected = None
    if settings.memory.embedding_model_id:
        for model in candidates:
            if model.id == settings.memory.embedding_model_id:
                selected = model
                break
    elif candidates:
        selected = candidates[0]
    if selected is None:
        return {
            "embedding_model_registered": False,
            "embedding_model_path_exists": False,
            "model_id": settings.memory.embedding_model_id,
            "path_basename": None,
        }
    resolved = selected.resolved_path(settings.home)
    return {
        "embedding_model_registered": True,
        "embedding_model_path_exists": resolved.exists(),
        "model_id": selected.id,
        "path_basename": resolved.name,
    }


def _vector_index_report(settings: Any) -> dict[str, Any]:
    # Health inspection is deliberately offline: it validates the persisted
    # bytes but never probes Runtime merely to render doctor output.
    from services.memory.vector_memory import VectorMemory

    inspector = VectorMemory(
        settings.vector_index_path,
        embedding=HashedTokenEmbedding(),
        initialize=False,
    )
    report = inspector.health()
    persisted_dimensions = report.get("active_dimensions")
    expected_dimensions = (
        HashedTokenEmbedding().dimensions
        if settings.memory.embedding_provider == "hashed-token"
        else persisted_dimensions
        if type(persisted_dimensions) is int
        else HashedTokenEmbedding().dimensions
    )
    report = inspector.health(
        configured_provider=settings.memory.embedding_provider,
        configured_dimensions=expected_dimensions,
    )
    legacy_metadata = settings.vector_index_path / "metadata.json"
    if report.get("active_provider") is None and legacy_metadata.is_file():
        try:
            import json

            raw = json.loads(legacy_metadata.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raw = {}
        if isinstance(raw, dict):
            provider = raw.get("provider")
            dimensions = raw.get("dimensions")
            if isinstance(provider, str):
                report["active_provider"] = provider
            if type(dimensions) is int:
                report["active_dimensions"] = dimensions
    report["path_basename"] = settings.vector_index_path.name
    report["persisted_provider"] = report.get("active_provider")
    report["persisted_dimensions"] = report.get("active_dimensions")
    return report


def _verify_runtime_embedding(settings: Any, model_id: str | None) -> dict[str, Any]:
    client = RuntimeClient(
        settings.runtime.url,
        timeout=settings.runtime.request_timeout_seconds,
        token=settings.runtime.token,
    )

    async def _probe() -> list[float]:
        return await client.embed("april memory doctor", model_id=model_id)

    try:
        vector = asyncio.run(_probe())
    except Exception as exc:
        return {"status": "error", "message": str(exc)[:240]}
    return {
        "status": "ok",
        "model_id": model_id,
        "dimensions": len(vector),
    }


def _print_memory_doctor(data: dict[str, Any]) -> None:
    table = Table(title="APRIL Memory Doctor")
    table.add_column("Field")
    table.add_column("Value")
    for key in (
        "status",
        "configured_embedding_provider",
        "active_vector_index_provider",
        "dimensions",
        "runtime_local_requested",
        "fell_back_to_hashed_token",
        "fallback_risk",
        "reindex_required",
        "embedding_model_id",
        "embedding_role_model_registered",
        "embedding_model_path_exists",
        "embedding_model_path_basename",
        "active_generation",
        "last_successful_reindex_at",
    ):
        table.add_row(key, str(data.get(key)))
    console.print(table)
    if data.get("reindex_required"):
        console.print(
            "[yellow]Index provider/dimension does not match the active provider; "
            "reindex is required.[/yellow]"
        )
        console.print("Next command:")
        # markup=False so command tokens are not parsed as Rich tags.
        console.print(f"  {data.get('reindex_command')}", markup=False)
    vector_index = data.get("vector_index", {})
    if isinstance(vector_index, dict) and vector_index.get("fallback_active"):
        console.print("[yellow]A validated recovery generation is active read-only.[/yellow]")
        console.print("Repair command:")
        console.print(f"  {data.get('repair_command')}", markup=False)
    if (
        isinstance(vector_index, dict)
        and vector_index.get("status") == "not_ready"
        and not data.get("reindex_required")
    ):
        console.print("Rebuild command:")
        console.print(f"  {data.get('reindex_command')}", markup=False)
    if data.get("setup_command"):
        console.print("Configure a runtime-local embedding model:")
        console.print(f"  {data.get('setup_command')}", markup=False)
