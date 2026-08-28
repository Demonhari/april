# LoRA canary safety design

Status: **implemented in software; disabled by default and target-Mac evidence
required**.

APRIL resolves the baseline adapter through its normal model-level pointer, but
LoRA canaries use a separately addressable Runtime instance. That instance owns
an immutable backend and binds base-model, adapter, and rollout-configuration
hashes. APRIL never switches the global adapter per request.

## Required Runtime architecture

Safe support requires both of these independently addressable identities:

- an immutable baseline identity binding model bytes, adapter bytes (if any),
  hashes, context/runtime parameters, and chat format;
- an immutable candidate identity binding the same fields to the candidate
  adapter and its verified hash.

Both identities must be loadable concurrently in one Runtime without shared
mutable adapter state, or the candidate must run in a separate authenticated,
loopback-only Runtime process. Each request must name its chosen identity.
Selection must be deterministic from a stable eligible-request identifier and
the rollout ID; retry and restart must select the same identity.

The implementation must enforce an Intel Mac memory budget before loading the
candidate. It must account for both model mappings, adapter allocations,
context/KV caches, process overhead, and a safety reserve. If concurrent loading
does not fit, the canary remains blocked; unloading the baseline or swapping a
global adapter is not an acceptable workaround.

## Activation, recovery, and audit requirements

Canary start and full activation require separate exact Level 4 approvals bound
to the rollout ID, immutable candidate identity, hashes, limits, and action.
Publication must use the existing two-phase state transition: prepare durable
database state, publish the pointer atomically, and commit the matching state.
Rollback must restore the exact previous identity, be idempotent, and never
leave baseline and candidate both marked active.

Startup reconciliation must cover interruption before publication, after
publication but before database commit, during rollback, and during candidate
unload. Hash-chained audit events must record preparation, publication,
assignment totals, rollback reason codes, reconciliation, and unload without
prompts, conversations, tool output, credentials, or raw model output.
Candidate unload must be explicit, bounded, retryable, and safe after Runtime or
process failure.

## Test strategy

Unit and fake-Runtime tests must cover immutable identity/hash validation,
deterministic assignment, excluded high-risk requests, concurrent baseline and
candidate requests, exact approvals, bounded turns/expiry, memory-budget
rejection, candidate-load failure, hard-failure rollback, threshold rollback,
both publication crash points, restart reconciliation, candidate unload,
idempotent rollback, exact prior restoration, audit-chain integrity, and
content-redacted evidence.

Real-model acceptance on the target Intel Mac additionally requires:

- both identities concurrently loaded or demonstrably isolated in separate
  authenticated Runtime processes;
- verified memory headroom and no unsafe sustained degradation;
- baseline-versus-candidate reviewed quality, structured-output, coding,
  latency, and failure evidence;
- deterministic bounded traffic under concurrency and restarts;
- observed load, unload, hard-failure rollback, crash recovery, and exact
  restoration;
- owner review and exact Level 4 canary and activation approvals.

When Runtime cannot prove isolated candidate capability, the safe result is a
typed unavailable/blocking reason and no canary traffic is selected. Real-model
and target-Mac acceptance remain operator evidence rather than claims made by
these software tests.
