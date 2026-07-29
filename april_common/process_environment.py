from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path

PROCESS_ENVIRONMENT_POLICY_VERSION = "phase4b-credential-id-v2"


class ProcessCategory(StrEnum):
    DAEMON = "daemon"
    CORE_API = "core_api"
    RUNTIME = "runtime"
    SENTINEL_VOICE = "sentinel_voice"
    JOB_WORKER = "job_worker"
    TOOL_WORKER = "tool_worker"
    RESTRICTED_COMMAND = "restricted_command"
    TEST_RUNNER = "test_runner"
    GIT = "git"
    REPOSITORY_INDEXING = "repository_indexing"
    DOCUMENT_PROCESSING = "document_processing"
    MODEL_VERIFICATION = "model_verification"
    BENCHMARKING = "benchmarking"
    VERIFICATION_SUBPROCESS = "verification_subprocess"
    CLI = "cli"


_BASE_KEYS = frozenset(
    {
        "PATH",
        "HOME",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "VIRTUAL_ENV",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUTF8",
        "PYTHONIOENCODING",
        "PYTHONDONTWRITEBYTECODE",
        "SYSTEMROOT",  # Windows compatibility; harmless when absent.
    }
)
_SERVICE_TOKEN_KEYS: dict[ProcessCategory, frozenset[str]] = {}
_CATEGORY_APRIL_KEYS: dict[ProcessCategory, frozenset[str]] = {
    ProcessCategory.DAEMON: frozenset(
        {
            "APRIL_ENVIRONMENT",
            "APRIL_JOB_WORKER_ENABLED",
            "APRIL_TOOL_WORKER_ENABLED",
        }
    ),
    ProcessCategory.CORE_API: frozenset(
        {
            "APRIL_ALLOWED_FILESYSTEM_ROOTS",
            "APRIL_API_HOST",
            "APRIL_API_PORT",
            "APRIL_AUDIT_PATH",
            "APRIL_DATABASE_PATH",
            "APRIL_ENVIRONMENT",
            "APRIL_JOB_WORKER_EXTERNAL",
            "APRIL_JOB_WORKER_ENABLED",
            "APRIL_LOGS_PATH",
            "APRIL_MEMORY_EMBEDDING_MODEL_ID",
            "APRIL_MEMORY_EMBEDDING_PROVIDER",
            "APRIL_RUNTIME_BACKEND",
            "APRIL_RUNTIME_URL",
            "APRIL_TOOL_WORKER_EXTERNAL",
            "APRIL_TOOL_WORKER_ENABLED",
            "APRIL_VECTOR_INDEX_PATH",
        }
    ),
    ProcessCategory.RUNTIME: frozenset(
        {
            "APRIL_ENVIRONMENT",
            "APRIL_RUNTIME_BACKEND",
            "APRIL_RUNTIME_HOST",
            "APRIL_RUNTIME_PORT",
            "APRIL_RUNTIME_PRELOAD_KEEP_LOADED",
        }
    ),
    ProcessCategory.SENTINEL_VOICE: frozenset({"APRIL_ENVIRONMENT"}),
    ProcessCategory.JOB_WORKER: frozenset(
        {
            "APRIL_ALLOWED_FILESYSTEM_ROOTS",
            "APRIL_AUDIT_PATH",
            "APRIL_DATABASE_PATH",
            "APRIL_ENVIRONMENT",
            "APRIL_LOGS_PATH",
            "APRIL_MEMORY_EMBEDDING_MODEL_ID",
            "APRIL_MEMORY_EMBEDDING_PROVIDER",
            "APRIL_RUNTIME_BACKEND",
            "APRIL_RUNTIME_URL",
            "APRIL_VECTOR_INDEX_PATH",
        }
    ),
    ProcessCategory.TOOL_WORKER: frozenset(),
    ProcessCategory.RESTRICTED_COMMAND: frozenset(),
    ProcessCategory.TEST_RUNNER: frozenset(),
    ProcessCategory.GIT: frozenset(),
    ProcessCategory.REPOSITORY_INDEXING: frozenset(),
    ProcessCategory.DOCUMENT_PROCESSING: frozenset(),
    ProcessCategory.MODEL_VERIFICATION: frozenset({"APRIL_RUNTIME_BACKEND"}),
    ProcessCategory.BENCHMARKING: frozenset({"APRIL_RUNTIME_BACKEND"}),
    ProcessCategory.VERIFICATION_SUBPROCESS: frozenset(
        {
            "APRIL_ALLOWED_FILESYSTEM_ROOTS",
            "APRIL_API_HOST",
            "APRIL_API_PORT",
            "APRIL_API_TOKEN",
            "APRIL_AUDIT_PATH",
            "APRIL_DATABASE_PATH",
            "APRIL_LOGS_PATH",
            "APRIL_RUNTIME_BACKEND",
            "APRIL_RUNTIME_HOST",
            "APRIL_RUNTIME_PORT",
            "APRIL_RUNTIME_PRELOAD_KEEP_LOADED",
            "APRIL_RUNTIME_TOKEN",
            "APRIL_RUNTIME_URL",
            "APRIL_VECTOR_INDEX_PATH",
        }
    ),
    ProcessCategory.CLI: frozenset(
        {
            "APRIL_API_HOST",
            "APRIL_API_PORT",
            "APRIL_ENVIRONMENT",
        }
    ),
}
_CREDENTIAL_IDENTIFIER_KEYS = frozenset(
    {
        "APRIL_CREDENTIAL_STORE",
        "APRIL_CREDENTIAL_FILE_PATH",
        "APRIL_API_CREDENTIAL_ID",
        "APRIL_RUNTIME_CREDENTIAL_ID",
        "APRIL_AUDIT_ANCHOR_CREDENTIAL_ID",
    }
)
_NETWORK_DENIED = frozenset(
    {
        ProcessCategory.TOOL_WORKER,
        ProcessCategory.RESTRICTED_COMMAND,
        ProcessCategory.TEST_RUNNER,
        ProcessCategory.GIT,
        ProcessCategory.REPOSITORY_INDEXING,
        ProcessCategory.DOCUMENT_PROCESSING,
    }
)
_PROXY_KEYS = frozenset(
    {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    }
)


def build_process_environment(
    category: ProcessCategory,
    *,
    source: Mapping[str, str] | None = None,
    april_home: Path | None = None,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a minimal category-specific child environment from an allowlist.

    Environment values are never logged or returned through diagnostics. Call
    :func:`allowed_environment_names` when safe name-only diagnostics are needed.
    """
    parent = source if source is not None else os.environ
    allowed = set(_BASE_KEYS)
    allowed.update(key for key in parent if key.startswith("LC_"))
    allowed.update(_SERVICE_TOKEN_KEYS.get(category, frozenset()))
    allowed.update(_CATEGORY_APRIL_KEYS.get(category, frozenset()))
    if category in {
        ProcessCategory.DAEMON,
        ProcessCategory.CORE_API,
        ProcessCategory.RUNTIME,
        ProcessCategory.SENTINEL_VOICE,
        ProcessCategory.JOB_WORKER,
        ProcessCategory.CLI,
    }:
        allowed.update(_CREDENTIAL_IDENTIFIER_KEYS)
    if april_home is not None:
        allowed.add("APRIL_HOME")
    if category in _NETWORK_DENIED:
        allowed.difference_update(_PROXY_KEYS)

    environment = {key: parent[key] for key in sorted(allowed) if key in parent}
    if april_home is not None:
        environment["APRIL_HOME"] = str(april_home.expanduser().resolve())
    for key, value in (overrides or {}).items():
        if key not in allowed:
            raise ValueError(
                f"Environment override {key!r} is not allowed for category {category.value}."
            )
        environment[key] = value
    return environment


def without_raw_credentials(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copy an environment while excluding APRIL's raw service credentials."""
    parent = source if source is not None else os.environ
    return {
        key: value
        for key, value in parent.items()
        if key not in {"APRIL_API_TOKEN", "APRIL_RUNTIME_TOKEN"}
    }


def allowed_environment_names(
    category: ProcessCategory,
    *,
    source: Mapping[str, str] | None = None,
    include_april_home: bool = True,
) -> tuple[str, ...]:
    environment = build_process_environment(
        category,
        source=source,
        april_home=Path("/") if include_april_home else None,
    )
    return tuple(sorted(environment))
