from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from apps.runner.verify import ModelBenchmark, RealModelVerifier
from april_common.settings import load_settings
from services.jobs.model_jobs import validate_registered_model


def _verification(home: Path, model_id: str) -> dict[str, Any]:
    settings = load_settings(root=home, legacy_credential_migration=True)
    model = validate_registered_model(settings, model_id)
    verifier = RealModelVerifier(
        home=home,
        model_path=model.path,
        inherit_process_group=True,
    )
    checks = verifier.run()
    return {
        "schema_version": 1,
        "report_type": "model_import_verification",
        "model_id": model_id,
        "passed": all(check.ok for check in checks),
        "checks": [
            {"name": check.name, "ok": check.ok, "status": check.status} for check in checks
        ],
        "measurements": {
            "load_time_seconds": verifier.load_time_seconds,
            "first_token_latency_seconds": verifier.first_token_latency_seconds,
            "generation_time_seconds": verifier.generation_time_seconds,
            "output_tokens": verifier.output_tokens,
            "tokens_per_second": verifier.tokens_per_second,
            "runtime_rss_bytes": verifier.runtime_rss_bytes,
        },
    }


def _benchmark(home: Path, model_id: str) -> dict[str, Any]:
    settings = load_settings(root=home, legacy_credential_migration=True)
    model = validate_registered_model(settings, model_id)
    benchmark = ModelBenchmark(
        home=home,
        model_path=model.path,
        prompt="Return a compact valid JSON object.",
        runs=settings.benchmark.runs,
        max_output_tokens=64,
        keep_loaded=False,
        inherit_process_group=True,
    )
    runs = benchmark.run()
    successful = [run for run in runs if run.ok]
    return {
        "schema_version": 1,
        "report_type": "model_benchmark",
        "model_id": model_id,
        "passed": len(successful) == len(runs) and bool(runs),
        "runs": [
            {
                "run_index": run.run_index,
                "ok": run.ok,
                "load_time_seconds": run.load_time_seconds,
                "first_token_latency_seconds": run.first_token_latency_seconds,
                "generation_time_seconds": run.generation_time_seconds,
                "output_tokens": run.output_tokens,
                "tokens_per_second": run.tokens_per_second,
                "unload_success": run.unload_success,
            }
            for run in runs
        ],
        "measurements_unavailable": [
            "prompt_processing_tokens_per_second",
            "peak_rss_bytes",
            "thermal_throttling",
            "fixture_quality",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--mode", required=True, choices=("verify", "benchmark"))
    args = parser.parse_args()
    try:
        payload = (
            _verification(args.home, args.model_id)
            if args.mode == "verify"
            else _benchmark(args.home, args.model_id)
        )
    except Exception:
        print(json.dumps({"error_code": f"model_{args.mode}_worker_failed"}, sort_keys=True))
        raise SystemExit(1) from None
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
