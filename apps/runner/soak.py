from __future__ import annotations

import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from apps.runner.verify import LauncherVerifier, _process_rss_bytes
from april_common.time import utc_now_iso


class SoakReport(BaseModel):
    schema_version: int = 1
    report_type: str = "soak"
    generated_at: str
    runtime_backend: str = "fake"
    real_model_verified: bool = False
    duration_seconds: float
    iterations: int = 0
    failures: list[str] = Field(default_factory=list)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    process_rss_bytes: int | None = None
    cycled_fake_models: bool = False
    summary: str = "degraded"


def write_soak_report(report: SoakReport, path: Path) -> Path:
    resolved = path.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return resolved


def _latency_summary(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {}
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {
        "min": round(min(samples), 2),
        "median": round(statistics.median(samples), 2),
        "p95": round(ordered[p95_index], 2),
        "max": round(max(samples), 2),
    }


def _timed(action: str, failures: list[str], samples: list[float], fn: Any) -> None:
    started = time.monotonic()
    try:
        fn()
    except Exception as exc:
        failures.append(f"{action}: {exc}")
    finally:
        samples.append((time.monotonic() - started) * 1000)


def run_fake_soak(
    home: Path,
    *,
    minutes: float,
    interval_seconds: float = 1.0,
    cycle_models: bool = False,
) -> SoakReport:
    verifier = LauncherVerifier(home=home)
    failures: list[str] = []
    latencies: list[float] = []
    iterations = 0
    duration_seconds = max(minutes * 60.0, 0.1)
    deadline = time.monotonic() + duration_seconds
    try:
        verifier._prepare()
        env = verifier._env()
        verifier.runtime = verifier._start(
            "services.april_runtime.server", env, verifier.runtime_log
        )
        verifier.api = verifier._start("services.api.server", env, verifier.api_log)
        verifier._wait_json(verifier.runtime_url + "/runtime/health")
        verifier._wait_json(verifier.api_url + "/health")
        with httpx.Client(
            base_url=verifier.api_url,
            headers={"Authorization": f"Bearer {verifier.api_token}"},
            timeout=10.0,
        ) as client:
            while True:
                iterations += 1
                _timed(
                    "core health",
                    failures,
                    latencies,
                    lambda: client.get("/health").raise_for_status(),
                )
                _timed(
                    "runtime models",
                    failures,
                    latencies,
                    lambda: client.get("/runtime/models").raise_for_status(),
                )
                _timed(
                    "chat",
                    failures,
                    latencies,
                    lambda: client.post(
                        "/chat",
                        json={"message": "Fake soak health check."},
                    ).raise_for_status(),
                )
                if cycle_models:
                    _timed(
                        "fake model load",
                        failures,
                        latencies,
                        lambda: client.post(
                            "/runtime/models/load",
                            json={"model_id": "april-brain"},
                        ).raise_for_status(),
                    )
                    _timed(
                        "fake model unload",
                        failures,
                        latencies,
                        lambda: client.post(
                            "/runtime/models/unload",
                            json={"model_id": "april-brain"},
                        ).raise_for_status(),
                    )
                if time.monotonic() >= deadline:
                    break
                time.sleep(max(interval_seconds, 0.1))
    except Exception as exc:
        failures.append(f"soak harness: {exc}")
    finally:
        rss = _process_rss_bytes(verifier.api.pid if verifier.api else None)
        verifier._stop()
        shutil.rmtree(verifier.temp, ignore_errors=True)
    return SoakReport(
        generated_at=utc_now_iso(),
        duration_seconds=duration_seconds,
        iterations=iterations,
        failures=failures,
        latency_ms=_latency_summary(latencies),
        process_rss_bytes=rss,
        cycled_fake_models=cycle_models,
        summary="pass" if iterations > 0 and not failures else "fail",
    )
