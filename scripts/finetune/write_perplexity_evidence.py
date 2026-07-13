from __future__ import annotations

import argparse
import json
from pathlib import Path

from april_common.settings import load_settings
from april_common.time import utc_now_iso
from services.evolution.adapters import sha256_file
from services.evolution.write_guard import EvolutionWriteGuard


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write APRIL LoRA perplexity evidence from already-measured local "
            "base/adapter perplexity scores."
        )
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-path", required=True, type=Path)
    parser.add_argument("--base-ppl", required=True, type=float)
    parser.add_argument("--adapter-ppl", required=True, type=float)
    parser.add_argument("--heldout-dataset", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    adapter_path = args.adapter_path.expanduser().resolve(strict=False)
    if not adapter_path.exists():
        raise SystemExit(f"adapter file does not exist: {adapter_path}")
    if args.base_ppl <= 0 or args.adapter_ppl <= 0:
        raise SystemExit("perplexity scores must be positive")

    settings = load_settings()
    output = args.output.expanduser()
    if not output.is_absolute():
        output = settings.home / output
    payload = {
        "schema_version": 1,
        "evidence_type": "lora_perplexity",
        "model_id": args.model_id,
        "adapter_path": str(adapter_path),
        "adapter_sha256": sha256_file(adapter_path),
        "base_perplexity": args.base_ppl,
        "adapter_perplexity": args.adapter_ppl,
        "heldout_dataset": args.heldout_dataset,
        "created_at": utc_now_iso(),
    }
    try:
        written = EvolutionWriteGuard(settings).write_text(
            output,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
    except PermissionError as exc:
        raise SystemExit("output must be under data/evolution/ or data/playbooks/") from exc
    print(str(written))


if __name__ == "__main__":
    main()
