from __future__ import annotations

import os
import resource
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from services.april_runtime.model_lifecycle import ModelLifecycle
from services.april_runtime.schemas import MAX_EMBED_BATCH_ITEMS, RuntimeHealth


@dataclass(frozen=True, slots=True)
class ProcessMemoryMetrics:
    rss_bytes: int | None
    peak_rss_bytes: int | None
    estimated: bool = True


MetricProvider = Callable[[], ProcessMemoryMetrics]


def runtime_health(
    lifecycle: ModelLifecycle,
    *,
    backend: str,
    request_id: str | None = None,
    metric_provider: MetricProvider | None = None,
    perf_profile: str | None = None,
) -> RuntimeHealth:
    models = lifecycle.list_models()
    missing = [model.id for model in models if model.missing_path]
    loaded = [model for model in models if model.state == "loaded"]
    metrics = (metric_provider or process_memory_metrics)()
    # The lifecycle's actual root backend is authoritative (the server passes the
    # configured backend string, but a test may inject a fake lifecycle).
    effective_backend = lifecycle.root_backend or backend
    simulated = effective_backend == "fake"
    has_backend_error = any(model.state == "error" for model in models)
    candidate_readiness = lifecycle.candidate_readiness()
    prefix_stats: dict[str, int] = {}
    for model in models:
        if not model.prefix_cache:
            continue
        for key in (
            "entries",
            "bytes",
            "lookups",
            "hits",
            "misses",
            "restores",
            "restore_failures",
            "saves",
            "evictions",
            "rejected_oversize",
            "attach_skipped",
        ):
            value = model.prefix_cache.get(key)
            if isinstance(value, int):
                prefix_stats[key] = prefix_stats.get(key, 0) + value
    # A working fake runtime is "ok" even though its configured GGUF paths do not
    # exist: it never loads them. Missing real-model paths stay informational in
    # ``missing_models`` so simulation is never mistaken for real readiness, and
    # genuine backend/model errors still degrade in both modes. A real backend
    # with missing files remains degraded as before.
    if simulated:
        status = "degraded" if has_backend_error else "ok"
    else:
        status = "degraded" if (missing or has_backend_error) else "ok"
    return RuntimeHealth(
        status=status,
        backend=effective_backend,
        simulated=simulated,
        models=models,
        missing_models=missing,
        request_id=request_id or str(uuid.uuid4()),
        loaded_model_count=len(loaded),
        active_requests=sum(model.active_requests for model in models),
        generation_error_count=sum(model.generation_errors for model in models),
        embedding_model_id=lifecycle.embedding_model_id(),
        batch_embedding_supported=True,
        batch_embedding_max_items=MAX_EMBED_BATCH_ITEMS,
        lifecycle_policy=lifecycle.policy_snapshot(),
        process_rss_bytes=metrics.rss_bytes,
        process_peak_rss_bytes=metrics.peak_rss_bytes,
        process_memory_estimated=metrics.estimated,
        lora_isolated_candidate_supported=bool(
            lifecycle.candidate_capability().get("supported", False)
        ),
        baseline_model_instance=str(candidate_readiness.get("baseline_model_instance"))
        if candidate_readiness.get("baseline_model_instance") is not None
        else None,
        candidate_instances=(
            list(candidate_items)
            if isinstance(candidate_items := candidate_readiness.get("candidate_instances"), list)
            else []
        ),
        candidate_integrity_state=str(
            candidate_readiness.get("candidate_integrity_state", "unknown")
        ),
        prefix_cache=prefix_stats or None,
        perf_profile=perf_profile,
    )


def process_memory_metrics() -> ProcessMemoryMetrics:
    rss = _psutil_rss()
    estimated = rss is None
    if rss is None:
        rss = _linux_proc_rss()
    peak = _peak_rss()
    return ProcessMemoryMetrics(rss_bytes=rss, peak_rss_bytes=peak, estimated=estimated)


def _psutil_rss() -> int | None:  # pragma: no cover - host-specific probe failures
    try:
        import psutil
    except Exception:
        return None
    try:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def _linux_proc_rss() -> int | None:  # pragma: no cover - host-specific probe failures
    statm = Path("/proc/self/statm")
    if not statm.exists():
        return None
    try:
        pages = int(statm.read_text(encoding="utf-8").split()[1])
    except Exception:
        return None
    return pages * os.sysconf("SC_PAGE_SIZE")


def _peak_rss() -> int | None:  # pragma: no cover - host-specific platform probe
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if value <= 0:
        return None
    if sys.platform == "darwin":
        return value
    return value * 1024
