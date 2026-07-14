# M15 — Local LoRA fine-tuning runbook (manual, CPU-only)

APRIL does **not** train models itself. This runbook documents the manual,
fully-local path from APRIL's exported dataset to a LoRA adapter served by
April Runtime, and states honestly which steps are automated and which are not.

## What APRIL automates

1. **Dataset export** (implemented, tested):

   ```bash
   run april evolve dataset export --name my-dataset
   # → data/evolution/datasets/my-dataset.jsonl
   ```

   The export is a reviewable JSONL file of chat prompt/response pairs, durable
   memories, and preference pairs (`prompt`, `chosen`, `rejected`) when a
   bad-rated reply has a real correction reply or a good-rated counterpart to
   the same prompt. Negative-feedback conversations remain excluded from chat
   rows; they contribute a preference row only when both sides exist. No chosen
   response is fabricated. Deleted/superseded/expired memories and
   sensitive-looking content (tokens, keys, passwords) are never exported.
   **Review the file manually before training on it.**

2. **Adapter serving and lifecycle bookkeeping** (implemented, tested against
   the fake/mocked backend): either set `adapter_path` on a model in
   `configs/models.yaml` as a manual override, or activate a versioned pointer
   under `data/evolution/adapters/` with `april evolve adapter activate`. The
   Runtime never opens the API SQLite database; adapter resolution is:
   explicit config `adapter_path` > active fenced pointer file > no adapter.
   A configured-or-pointer-selected missing adapter file fails the load with an
   actionable error instead of silently serving the base model.

   ```yaml
   models:
     brain:
       id: april-brain
       path: models/granite3.3-2b-q4_k_m.gguf
       adapter_path: models/adapters/april-brain-lora.gguf
       ...
   ```

3. **Evidence JSON writing from already-measured scores** (implemented): APRIL
   ships a small helper that records operator-provided base/adapter perplexity
   numbers and the adapter SHA-256. It does not train and does not calculate
   perplexity for you.

   ```bash
   .venv/bin/python scripts/finetune/write_perplexity_evidence.py \
     --model-id april-brain \
     --adapter-path /absolute/path/april-brain-lora.gguf \
     --base-ppl 12.4 \
     --adapter-ppl 11.9 \
     --heldout-dataset data/evolution/datasets/my-heldout.jsonl \
     --output data/evolution/adapters/evidence/april-brain.json
   ```

## What you must do manually (not automated, not verified here)

Training does not run inside APRIL: there is no local training loop, no GPU
assumption, and no network download. On an Intel MacBook Pro the practical
path is CPU-only and slow — budget hours, not minutes, even for small models.

1. **Convert the dataset** to your trainer's format. Chat rows look like
   `{"type": "chat", "prompt": ..., "response": ...}`; preference rows use
   `{"type": "preference", "prompt": ..., "chosen": ..., "rejected": ...}`.
   Most supervised trainers want a `{"text": "<prompt>\n<response>"}` or
   chat-template format, while preference trainers have their own pair schema.
   Write your own small converter; keep it outside APRIL's runtime.

2. **Train a LoRA adapter** with an external, locally-installed tool.
   Two realistic CPU-only options:
   - `llama.cpp` finetune tooling (`llama-finetune`, from the same llama.cpp
     checkout family as your GGUF), which trains directly against a GGUF base
     model and can emit a GGUF LoRA;
   - PyTorch + PEFT on the original HF checkpoint, followed by
     `llama.cpp/scripts/convert_lora_to_gguf.py` to produce a GGUF LoRA.

3. **Place the adapter locally** (Git-ignored), e.g.
   `models/adapters/april-brain-lora.gguf`.

4. **Measure held-out perplexity locally.** Use your training/eval tooling to
   measure base-model perplexity and adapter perplexity on held-out personal
   data. APRIL never synthesizes these scores; activation requires
   `adapter_ppl <= base_ppl`.

5. **Write the evidence JSON** with the helper above, then verify the candidate
   adapter like any other real-model change:

   ```bash
   .venv/bin/python scripts/finetune/write_perplexity_evidence.py \
     --model-id april-brain \
     --adapter-path /absolute/path/april-brain-lora.gguf \
     --base-ppl BASE_PPL \
     --adapter-ppl ADAPTER_PPL \
     --heldout-dataset data/evolution/datasets/my-heldout.jsonl \
     --output data/evolution/adapters/evidence/april-brain.json
   run april model doctor
   run april verify --all-configured-models --require-real-model \
     --candidate-adapter-model-id april-brain \
     --candidate-adapter-path /absolute/path/april-brain-lora.gguf \
     --report data/verification/mac-readiness.json
   ```

6. **Activate or roll back the pointer.** In production, activation also
   requires the fresh real-model verification report above to show the same
   adapter SHA-256 was loaded. Rollback flips only the active pointer.

   ```bash
   april evolve adapter activate april-brain /absolute/path/april-brain-lora.gguf \
     --evidence data/evolution/adapters/evidence/april-brain.json \
     --verification-report data/verification/mac-readiness.json
   april evolve adapter list --model-id april-brain
   april evolve adapter rollback april-brain
   ```

## Current blockers (why this is not one command)

- **No training dependency ships with APRIL.** Installing llama.cpp finetune
  binaries or PyTorch/PEFT is a manual, local decision; APRIL never installs
  packages or downloads tools/models.
- **No GPU on the target Intel MacBook Pro.** CPU LoRA training works but is
  slow; that trade-off is yours to accept per run.
- **Perplexity measurement is manual.** APRIL can record evidence and enforce
  the activation gate, but it does not run the evaluation or invent scores.
  Missing evidence blocks activation with the next command to run.
- **Production activation needs a real-model report.** In `APRIL_ENV=production`,
  a fresh `--all-configured-models --require-real-model` report must have loaded
  the same adapter hash. Fake verification never proves adapter quality.
- **`llama-cpp-python` LoRA support varies by version.** `lora_path` is
  honoured by the pinned `>=0.2.90` line, but confirm your installed wheel
  supports the adapter format you produced (GGUF LoRA vs legacy ggml LoRA).

Until you complete steps 1–6 on this Mac, LoRA serving is *wired but
unverified* — treat an adapter the same way as any other real-model
configuration that has not yet passed `--require-real-model` verification.
