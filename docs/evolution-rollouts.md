# Evolution rollout operator guide

Phase 4B is implemented but disarmed. The stock configuration keeps all of the
following false:

```yaml
evolution:
  enabled: false
  rollout_enabled: false
  automatic_candidate_creation: false
  canary_enabled: false
  automatic_promotion: false
```

Enabling one switch does not enable another. Dreamer remains disarmed, rollout
candidates are created only by an owner command, canary needs its own exact
Level 4 approval, and full activation needs a different exact Level 4 approval.
Good metrics never promote a candidate automatically.

## What is implemented

- Schema 21 durable rollout, assignment, and safe event records.
- Prompt-overlay A/B shadow evaluation through reviewed local cases.
- Durable `evolution_shadow` jobs with progress, cancellation, and lease
  recovery.
- Stable, bounded, restart-safe prompt canary selection.
- Low-risk eligibility filtering and safe aggregated outcome monitoring.
- No-regression and hard-failure thresholds.
- Two-phase prompt publication, exact prior-version restoration, idempotent
  rollback, startup reconciliation, and hash-chained audit records.
- CLI status, create, shadow, approval request, canary, promote, cancel, and
  rollback commands.
- Offline, API, and verification readiness reporting.
- An explicit `lora_canary_unsupported` failure. APRIL never switches a global
  LoRA pointer per request.

## What automated tests prove

Temporary-database and fake-evaluator tests prove state/concurrency rules,
minimum samples, no-regression gates, exact approvals, deterministic selection,
high-risk exclusion, cancellation, both publication interruption points,
automatic rollback, exact restoration, audit integrity, redaction, safe
defaults, and LoRA fail-closed behavior. They require no network, GGUF, audio
device, native model binding, or model download.

## What still requires real evidence

A real rollout needs an owner-reviewed dataset and local Runtime models,
including a distinct local judge for reviewed behavioral cases. The target
Intel Mac must produce real baseline/candidate quality, structured-output,
tool-selection, coding-test, latency, Runtime stability, memory, and thermal
evidence. Owner approval is required for canary and again for activation.

No real shadow, prompt canary, LoRA, GGUF, voice, or Intel Mac rollout is
claimed by the repository tests. LoRA cannot proceed at all until Runtime can
load a candidate as a separate immutable model identity alongside baseline.

## Recovery

Run:

```console
run april readiness
run april evolve rollout list --json
run april evolve rollout show ROLLOUT_ID --json
run april evolve rollout rollback ROLLOUT_ID --reason operator_recovery
run april audit verify
```

Restarting Core runs reconciliation before readiness. An incomplete
publication, expired undersampled canary, missing/tampered artifact, or
pointer/database mismatch fails readiness and attempts exact rollback. If the
previous artifact is itself unavailable, APRIL reports
`rollback_previous_unavailable` and remains unhealthy rather than guessing.
