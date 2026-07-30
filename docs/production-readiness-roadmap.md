# Production-readiness roadmap

This roadmap keeps APRIL offline, local-first, and disabled-by-default for
training, Dream Cycle, speaker gating, and sensitive-memory encryption. Fake
backends remain useful for software verification, but their reports are not
evidence of real model, microphone, thermal, signing, or notarization success.

## Verification status matrix

| Area | Status | Evidence still required |
|---|---|---|
| Core API, SQLite serialization, approvals, audit, backup/restore, deterministic routing, retrieval, and durable jobs | Implemented and tested | Normal CI and local regression suite |
| Tool Worker Seatbelt profiles and production fail-closed policy | Implemented and tested | Target-Mac socket-denial integration where `sandbox-exec` is operational |
| Durable staged model import and inactive registration | Implemented and tested | A separately supplied local GGUF for a real import, verification, and benchmark |
| Real GGUF load/chat/stream, prompt-eval timing, sustained performance, and setup comparison | Implemented but target-Mac verification required | Local model artifact and genuine real-runtime execution |
| Voice endpointing and optional voice adapters | Implemented and tested | Live microphone and speaker checks require those devices and local voice artifacts |
| Phase 4B prompt rollout state machine, shadow jobs, bounded canary, exact approvals, monitoring, reconciliation, and rollback | Implemented and fake/unit tested; disabled by default | Real reviewed-case A/B evidence with local GGUFs, owner approvals, and target-Intel-Mac canary evidence |
| Phase 4B LoRA canary | Explicitly blocked as unsupported | Runtime support for a separately loaded immutable candidate model identity; global adapter switching is not accepted |
| Production `.app` structure and release exclusions | Implemented and tested | Target-Mac bundle validation |
| Developer ID signing, notarization, stapling, and Gatekeeper | Implemented but target-Mac verification required | Real Apple signing identity, notary Keychain profile, and genuine Apple service execution |

Skipped hardware, model, voice, thermal, signing, or notarization checks remain
unverified; they are never counted as passed.

## Remaining operator actions

The implementation status and operating status are separate:

| Status | Meaning |
|---|---|
| Implemented in code | The guarded local workflow and its automated fake-backend tests exist. |
| Configured | The operator has supplied reviewed local paths and explicitly selected the intended provider or feature settings. |
| Verified on real hardware | The real artifact and target Intel Mac completed the applicable live command and produced genuine evidence. |
| Unavailable or disabled | A required local artifact, reviewed command, credential, or explicit feature enablement is absent. No success is inferred. |

The remaining operator work must be performed manually on the target Intel Mac:

1. Import the reviewed local Brain, coding, and reading GGUF files. APRIL does
   not download them and an import does not load, activate, or select a model.
2. Import and register `qwen3-4b Q4_K_M` with `role: reasoning`.
3. Import and register `nomic-embed-text-v1.5 Q8` with `role: embedding`, then
   explicitly change the embedding provider to Runtime-local and run
   `run april memory reindex --wait`.
4. Configure the local Whisper, Piper, and wake-word artifacts. Keep speaker
   verification optional and disabled until its configured model passes live
   verification.
5. Run the complete two-turn voice verification and the
   specialist-versus-shared-model benchmark on the target Intel Mac. Fake
   Runtime results are not production evidence.
6. Configure and review the fine-tuning trainer and evaluator commands before
   explicitly enabling fine-tuning.
7. Build, sign, notarize, and staple the application on a real macOS machine
   with the operator's Apple credentials.

Until each prerequisite is supplied, its status is **unavailable or disabled**;
configuration alone is not real-hardware verification.

## Durable model, training, and Dream jobs

`model_import` accepts an exact-approved absolute local source path, stages and
hashes the bytes, atomically publishes the GGUF, and registers it as an inactive,
low-priority candidate. It never downloads, loads, selects, or activates the
model. `model_import_verification` and `model_benchmark` accept only a model ID already
present in `configs/models.yaml`. APRIL resolves the registered path, verifies
containment and GGUF identity, and runs the existing real-model verifier in a
bounded process group. Neither job downloads, selects, or activates a model.

`finetune` is available only when `finetune.enabled` is reviewed and enabled.
The trainer and evaluator are exact executable paths plus argument templates
from `configs/april.yaml`; job payloads cannot supply commands. Training requires
an exact level-4 approval, has bounded resources/output/runtime, and is
non-restart-safe while the external trainer runs. Recovery marks an interrupted
attempt and requires an explicit retry. Cancellation terminates the restricted
process group. Durable acceptance, its initial event, and one-time approval
consumption are atomic. Results are registered only as inactive adapter
candidates.

`dream_cycle` remains unavailable unless evolution is explicitly enabled. It is
not automatically retryable. Candidate creation, rollout execution, canary
traffic, and automatic promotion are separately disabled by default. Phase 4B
does not let a production prompt or adapter activation use the legacy direct
activation path.

## Phase 4B rollout safety

Schema 21 adds durable rollout, assignment, and safe-event records. Prompt
overlays follow:

`candidate → shadow_pending → shadow_running → shadow_passed →`
`canary_pending_approval → canary_running → canary_passed →`
`activation_pending_approval → active`.

Terminal outcomes are `failed`, `cancelled`, `rolled_back`, and `rejected`.
Every mutation uses the existing serialized SQLite transaction path and an
optimistic version. Records bind candidate, baseline, configuration, reviewed
dataset, and shadow evidence hashes. Stored metrics contain only counters,
rates, bounded latency values, reason codes, and booleans; prompts,
conversations, generations, tool output, audio, and secrets are excluded.

`evolution_shadow` runs the existing reviewed-case A/B evaluator as a
restart-safe durable job. It gives the evaluator only the Runtime chat client,
not a tool executor. Training loss or perplexity is never sufficient evidence.
The same immutable cases are run for baseline and candidate, minimum sample
counts and no-regression gates are enforced, and shadow never changes an
active pointer.

Prompt canary traffic requires an exact Level 4 approval. Selection is a stable
hash of rollout ID and request ID, bounded by fraction, eligible-turn count, and
expiry. Voice, deep/council/high-stakes reasoning, write-capable agents,
approval-requiring interactions, repository/database writes, external or
destructive operations, and background evolution are excluded. Candidate
answers are ordinary responses only for selected eligible canary requests;
shadow answers are never user-visible.

The current Runtime binds a LoRA adapter to the globally loaded model. It does
not provide a second immutable candidate model identity for safe concurrent
routing. Phase 4B therefore reports `lora_canary_unsupported`, refuses LoRA
canary and promotion, and never toggles the global adapter per request.

Canary and newly-active prompt outcomes feed safe aggregate monitoring.
Integrity failure, Runtime failure, hard failure, regression threshold,
insufficient expired samples, or pointer/database disagreement triggers
idempotent rollback to the exact prior artifact. Prompt publication has a
durable prepared/published/finalized protocol; startup reconciliation
compensates an interruption on either side of publication. Rollback and
reconciliation emit hash-chained audit events.

Operator flow:

```console
# Explicitly review and enable evolution.enabled, evolution.rollout_enabled,
# and evolution.canary_enabled first. Automatic creation/promotion stay false.
run april evolve rollout create --type prompt_overlay --target-id general_agent \
  --candidate-id REVIEWED_ID --candidate-path data/evolution/candidates/FILE.overlay.txt
run april evolve rollout shadow-start ROLLOUT_ID
run april jobs status JOB_ID
run april evolve rollout approval-request ROLLOUT_ID --stage canary
run april approve APPROVAL_ID
run april evolve rollout canary-start ROLLOUT_ID --approval-id APPROVAL_ID
run april evolve rollout status ROLLOUT_ID
run april evolve rollout approval-request ROLLOUT_ID --stage activation
run april approve APPROVAL_ID
run april evolve rollout promote ROLLOUT_ID --approval-id APPROVAL_ID
run april evolve rollout rollback ROLLOUT_ID --reason operator_rollback
```

Creating an approval is an explicit owner command. Neither canary passage nor
good metrics creates, approves, consumes, or performs final promotion
automatically. `run april readiness`, Core `/readiness`, and `run april verify`
report disabled rollout state as disabled, while unsafe incomplete transitions
are readiness failures.

Automated tests use temporary SQLite databases, fake shadow evaluators, and no
GGUFs. They cover the state machine, approvals, bounds, deterministic selection,
concurrency, interruption, reconciliation, aggregate evidence, automatic
rollback, exact restoration, audit-chain validity, defaults, and explicit LoRA
blocking. They are not evidence that a real prompt, LoRA, GGUF, voice path, or
Intel Mac canary passed.

Fine-tune workflow:

```console
run april finetune doctor
run april finetune plan --dataset /reviewed/local/dataset.jsonl --base-model-id april-brain
run april finetune --plan-id PLAN_ID --approval-id APPROVAL_ID
run april finetune status JOB_ID
run april finetune cancel JOB_ID
```

Planning validates strict JSONL row schemas, redacts likely credentials and
sensitive paths, creates deterministic disjoint train/evaluation splits, and
writes an immutable review manifest containing dataset, configuration, base
model, trainer, and evaluator hashes. An evaluator failure produces no invented
perplexity value. The completed job prints separate verification and activation
commands; it never changes an active adapter pointer.

## Reasoning, embedding, and Intel Mac setup

APRIL does not download these artifacts. After obtaining and reviewing local
files, import them explicitly:

```console
run april model import --role reasoning --id qwen3-4b-reasoning --name "Qwen3-4B Q4_K_M" --path /LOCAL/Qwen3-4B-Q4_K_M.gguf --sha256 EXPECTED_SHA256
run april model import --role embedding --id nomic-embed-text-v1.5 --name "nomic-embed-text-v1.5 Q8" --path /LOCAL/nomic-embed-text-v1.5.Q8_0.gguf --sha256 EXPECTED_SHA256
run april memory reindex --wait
```

Readiness distinguishes an unregistered reasoning artifact, an unverified
registered artifact, hashed-token embeddings, a missing runtime-local embedding
model, a semantic model awaiting reindex, and a verified semantic generation.

On the first bootstrap of an Intel macOS host, APRIL selects
`intel_macbook_cpu_low` exactly once only if automatic selection is not
suppressed, there is no prior profile selection, and there is no manual
per-model runtime override. Evidence is recorded under local setup data.
`--no-auto-profile` suppresses this behavior; `--apply-profile` remains the
explicit override. Readiness and status never modify configuration.

Compare a locally registered shared model with the current specialists:

```console
run april model compare-setups --shared-model-id LOCAL_SHARED_MODEL_ID --wait
```

The job result excludes prompts, generations, source files, patches,
credentials, and absolute paths. Versioned offline routing, strict-JSON,
coding, context, lifecycle, and sustained-performance fixtures are identical
for both setups. Coding fixtures execute only through Tool Worker. Fake Runtime
results are labelled simulated and cannot produce a production
recommendation. Direct thermal state remains unavailable unless the platform
provides real evidence; sustained degradation is only a proxy. APRIL makes no
automatic model/profile selection. Live Intel Mac evidence must be generated
on the target machine.

The production recommendation gates require the complete installed fixture set
with the same hash for every setup and real (non-simulated) measurements for
routing accuracy, strict-JSON first-pass and final reliability, coding pass
rate, context reliability, first-token latency, generation and prompt
throughput, peak memory, load/unload reliability, load/switch overhead, and
sustained degradation. Defaults require routing ≥ 0.80, both JSON reliability
scores ≥ 0.90, coding ≥ 0.80, sustained degradation ≤ 0.15, and no required
metric regression beyond 15% versus the specialist setup. Missing fixtures or
metrics produce `insufficient_evidence`; a failed shared benchmark or absolute
gate produces `comparison_failed`; relative regressions produce
`manual_review_required`; only a complete passing real comparison is
`recommended`. The result is advisory and is never applied automatically.

## Live speaker verification

Speaker verification remains optional and never blocks push-to-talk, ordinary
voice conversation, or wake-word readiness while the gate is off.

```console
run april voice verify-speaker-live --report data/verification/speaker-live.json
run april voice speaker-gate enable-soft --report data/verification/speaker-live.json
run april voice speaker-gate disable
```

The live command requires the configured local ONNX model, ONNX Runtime,
reviewed enrollment samples, explicit confirmation, and a fresh recording.
Reports contain numeric similarities and fixture outcomes only. Raw audio is
deleted unless `--debug-capture` is explicitly selected. Enabling the soft gate
requires a fresh, successful, configuration-matched report; disable is
immediate. Mock tests are not live acceptance evidence.

## Locked dependencies and release exclusion

`uv.lock` is generated by uv from `pyproject.toml` and covers base, development,
and optional platform extras. Refresh and verify it with:

```console
uv lock
uv sync --locked --extra dev
uv lock --check
scripts/verify_locked_install.sh
```

Hardware/runtime, voice, desktop, and security dependencies remain optional.
Sensitive-memory AES-GCM uses the optional `security` extra; normal APRIL
execution never installs it.

Validate any release ZIP before distribution:

```console
run april package validate-release-zip dist/APRIL.zip
```

Validation fails on virtual environments, caches, databases, GGUF files,
adapters/keys/credentials, audio recordings, and generated verification data.

## Production macOS packaging

The existing development wrapper remains unsigned and separate. The production
path creates a stable `local.april.assistant` bundle with a microphone usage
description, minimum macOS version, optional `.icns`, deterministic metadata,
and hardened-runtime-compatible entitlements:

```console
run april package build --output dist/APRIL.app --version 0.1.0
run april package validate dist/APRIL.app
run april package sign dist/APRIL.app --identity "Developer ID Application: YOUR NAME (TEAMID)"
run april package verify-signature dist/APRIL.app
run april package archive dist/APRIL.app --output dist/APRIL.zip
run april package notarize-submit dist/APRIL.zip --keychain-profile APRIL_NOTARY
run april package notarize-status SUBMISSION_ID --keychain-profile APRIL_NOTARY
run april package staple dist/APRIL.app
run april package gatekeeper dist/APRIL.app
```

The notary profile must already have been created by the operator in Keychain.
APRIL stores no Apple credentials. Commands report success only when the real
Apple executable returns success. Optional owner launch-at-login:

```console
run april package launch-agent install dist/APRIL.app
run april package launch-agent remove
```

## Optional sensitive-memory encryption

This feature is disabled by default. Provision a Keychain key, review and set
`memory.sensitive_encryption_enabled: true`, and mark an individual memory write
as sensitive:

```console
run april security memory-encryption provision
run april security memory-encryption rotate
```

The versioned AES-GCM envelope authenticates content against its memory ID.
Ciphertext stays in SQLite and backups; raw keys never enter SQLite, reports,
logs, or child environments. Encrypted content is intentionally omitted from
FTS and vector indexes while non-sensitive metadata remains searchable. Missing
keys return an explicit unavailable marker, and corrupt ciphertext is never
silently deleted or overwritten. Rotation stages both keys, re-encrypts inside
the existing cross-process SQLite write transaction, then commits the new
Keychain keyring.

## Threat boundaries

These changes protect against accidental credential/dataset inclusion, arbitrary
training commands in job payloads, incomplete trainer recovery, leaked raw
speaker audio by default, casual reading of explicitly encrypted SQLite fields,
release archives containing local artifacts, and accidental activation of
models or adapters.

They do not protect a running process or logged-in account already controlled by
an attacker, Keychain compromise, malicious reviewed trainer/evaluator binaries,
screen/keyboard capture, firmware attacks, or disclosure through content the
operator deliberately decrypts and shares. Field encryption does not hide
non-sensitive metadata or access patterns. Signing/notarization establishes
Apple distribution identity and platform checks; it is not a claim that model
behavior is safe.
