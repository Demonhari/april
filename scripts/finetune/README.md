# M15 — Guided local LoRA fine-tuning

APRIL provides a reviewed, durable fine-tuning workflow around explicitly
configured local trainer and evaluator executables. Fine-tuning is disabled by
default. APRIL does not ship, install, or download a trainer, evaluator, model,
dataset, or adapter, and it never activates a resulting adapter automatically.

## Guided workflow

Check whether fine-tuning is enabled and whether both reviewed executables are
configured and executable:

```console
run april finetune doctor
```

Create an immutable plan from a reviewed local JSONL dataset:

```console
run april finetune plan \
  --dataset /reviewed/local/dataset.jsonl \
  --base-model-id april-brain
```

Planning validates strict chat, preference, and memory row schemas; rejects
oversized, malformed, sensitive-location, and out-of-scope input; redacts
likely credentials and sensitive paths; and creates deterministic, disjoint
training and evaluation splits. The plan records hashes for the normalized
dataset, both splits, reviewed configuration, base model, trainer, and
evaluator. It also creates a pending exact level-4 approval.

After reviewing the manifest and approval, launch exactly that plan:

```console
run april finetune --plan-id PLAN_ID --approval-id APPROVAL_ID
```

The launch revalidates the immutable plan and exact approval, then atomically
creates the durable job, its submitted event, and consumes the approval. An
exact replay returns the original job. A changed plan, hash, adapter candidate,
owner, or scope fails closed.

Inspect or cancel the durable job:

```console
run april finetune status JOB_ID
run april finetune cancel JOB_ID
```

## What the durable job runs

APRIL launches only the exact local trainer and evaluator executable paths and
argument templates reviewed in `configs/april.yaml`. Job payloads cannot
provide executable paths or commands. Child processes run with bounded time,
output, resources, filesystem access, and denied network access. APRIL never
installs a trainer or evaluator and never downloads their dependencies.

The trainer receives the hash-verified base model and deterministic training
and evaluation paths. Its output must be a local GGUF adapter candidate.
APRIL then invokes the configured evaluator for both the base model and
candidate. Each successful evaluator invocation must return a finite, positive
perplexity value in its final JSON output.

APRIL never invents perplexity results. A missing value, invalid value,
non-zero exit, cancellation, timeout, hash mismatch, or incomplete evaluation
fails the job and cannot create passing evidence.

On successful training and evaluation, APRIL writes hash-bound evidence and
registers the adapter as an `inactive_candidate`. Metric eligibility requires
candidate perplexity no worse than base perplexity, but that result alone never
activates the adapter.

## Verification and activation remain separate

The launch output provides the next verification and activation commands. The
candidate must still pass reviewed evidence and real-model verification,
including verification that the same adapter bytes were loaded. Production
activation requires a fresh qualifying real-model verification report.

Activation is always an explicit operator action:

```console
run april verify --all-configured-models --require-real-model \
  --candidate-adapter-model-id april-brain \
  --candidate-adapter-path data/evolution/adapters/candidates/CANDIDATE.gguf

run april evolve adapter activate april-brain \
  data/evolution/adapters/candidates/CANDIDATE.gguf \
  --evidence data/evolution/adapters/evidence/PLAN_ID.json \
  --verification-report data/verification/mac-readiness.json
```

The existing manual `write_perplexity_evidence.py` helper remains available
for adapters trained and evaluated outside the guided job. Operator-provided
numbers are recorded as supplied; the helper does not measure them.

Dream Cycle and autonomous evolution remain disabled by default. Neither can
enable fine-tuning, choose a trainer, launch a fine-tune job, approve one,
activate an adapter, or bypass the evidence and real-model gates.
