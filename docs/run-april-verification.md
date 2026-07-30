# Run APRIL Verification

APRIL release checks should include these local launcher gates:

Phase 4A also verifies schema-19 transitions and lease recovery, Job Worker
completion, the Tool Worker self-check and socket permissions, restricted
process timeout/output behavior, and service-token isolation. Unix-socket tests
may need an offline rerun outside a sandbox that denies local socket binding.

Phase 4B adds local credential, audit-chain, and SQLite lifecycle gates:

```bash
run april security credentials migrate
run april verify
run april audit verify
run april audit verify --json
run april database check
run april database check --full
run april database backup --output /private/backups/april-current.april
run april database restore --input /private/backups/april-current.april \
  --stop-services
```

`run april verify` without a model/fake/workflow flag is the bounded local
security and integrity report. Full SQLite `integrity_check` remains explicit.

```bash
run april config validate
run april config inspect
run april verify --fake
run april verify --soak --fake --minutes 10
run april verify --workflow
run april verify --target-mac
run april verify --all-configured-models --require-real-model --report data/verification/mac-readiness.json
run april verify --workflow --real-model --report data/verification/workflow-real.json
run april setup models --brain /absolute/path/granite.gguf --coding /absolute/path/qwen-coding.gguf --reading /absolute/path/qwen-reading.gguf --dry-run
run april setup voice --whisper-binary /path/to/whisper.cpp/main --whisper-model /path/to/ggml-base.en.bin --piper-binary /path/to/piper --piper-model /path/to/voice.onnx --dry-run
run april setup app-stub
run april model doctor
run april memory doctor --json
run april model profile list
run april status
run april stop
run april --fake ask "April, plan my work today."
run april --fake --oneshot ask "April, plan my work today."
run april model load april-brain --fake
run april model unload april-brain --fake
run april reminder create "stand up" --due-at 2026-06-21T09:00:00Z --fake
run april reminder list --fake
run april task list --fake
run april voice health --fake
run april voice doctor --fake
run april voice verify-live --report data/verification/voice-live.json
run april voice verify-wake-live --report data/verification/wake-live.json
run april voice verify-conversation-live \
  --report data/verification/voice-conversation-live.json
run april memory doctor
run april eval brain --fake
```

## Exact Target Mac Order

Run target-Mac setup and real verification in this order:

1. `run april readiness`
2. `run april setup bootstrap`
3. `run april setup tokens` if bootstrap reports token warnings
4. Confirm bootstrap's selected profile. On Intel macOS first run it applies
   `intel_macbook_cpu_low` exactly once unless suppressed, previously selected,
   or blocked by manual model/runtime overrides. Use
   `run april model profile apply intel_macbook_cpu_low` only when an explicit
   override is intended.
5. Validate with `run april setup models ... --dry-run`, then import each role with exact-approved `run april model import ... --sha256 EXPECTED_SHA256`
6. `pip install -e '.[runtime]'`
7. `run april verify --all-configured-models --require-real-model --report data/verification/mac-readiness.json`
8. `run april verify --workflow --real-model --report data/verification/workflow-real.json`
9. Optional voice setup/doctor/live verification:
   `run april setup voice --whisper-binary /path/to/whisper.cpp/main --whisper-model /path/to/ggml-base.en.bin --piper-binary /path/to/piper --piper-model /path/to/voice.onnx --dry-run`,
   `run april voice doctor`,
   `run april voice verify-live --report data/verification/voice-live.json`

Blank API credentials never authenticate. Production resolves API and Runtime
credentials from macOS Keychain and fails closed if Keychain is unavailable.
Legacy plaintext values are detected and must be moved with
`run april security credentials migrate`; token values are never printed.

`run april setup bootstrap` provisions secure credentials when the selected
platform/store is available and writes only non-secret store identifiers to
`.env`. JSON output redacts local absolute paths by default; use `--show-paths`
only when a local operator needs exact paths. The setup shell scripts use
`constraints-dev.txt` for reproducible base/dev editable installs and still do
not use sudo, Homebrew, model downloads, or automatic voice/runtime setup.

Project workflow smoke:

```bash
bash scripts/smoke_project_workflow.sh
```

Real GGUF smoke verification never downloads models. It skips with exit 0 when
no model path is provided:

```bash
run april verify --real-model
```

To run it, provide a local GGUF path:

```bash
APRIL_TEST_GGUF_PATH=/absolute/path/to/small-local-model.gguf run april verify --real-model
APRIL_TEST_GGUF_PATH=/absolute/path/to/small-local-model.gguf run april verify --workflow --real-model
APRIL_TEST_GGUF_PATH=/absolute/path/to/small-local-model.gguf run april verify --workflow --real-model --report data/verification/workflow-real.json
run april eval brain --real-model /absolute/path/to/small-local-model.gguf
run april model benchmark REGISTERED_MODEL_ID --wait
run april verify --target-mac /absolute/path/to/small-local-model.gguf --require-real-model
```

The real verifier starts isolated Runtime and Core API services on loopback
ports with a temporary Runtime token, loads the supplied GGUF through
`llama-cpp-python`, runs chat and streaming checks, unloads the model, confirms
the model state, and stops both services.

The real verifier reports load time, first token latency when streaming emits a
token, total generation time, output tokens, tokens/sec, context size, backend
settings, prompt path diagnostics, unload success, and Runtime RSS when the OS
reports it. If `llama-cpp-python` is missing, install the local runtime extra:

```bash
pip install -e '.[runtime]'
```

`run april verify --workflow --real-model` is a separate daily-use workflow
report, not the multi-model readiness report. It uses only verifier temporary
files/repos and checks runtime health, Core API health, non-fallback real
planning with `BrainDecision` validation, a `reading_agent` request, reminder
create/list, memory write/search, document indexing/search, temporary project
registration, read-only coding analysis, code-write approval creation, approval
denial, external/system action denial, and voice health/doctor status only. It
does not record audio, play audio, open the microphone, require wake-word models,
modify user repos, or send external requests. `--timeout` and
`--max-output-tokens` are passed into the real workflow verifier and may be
recorded as safe verifier settings in the workflow report.

Target-Mac validation is a local checklist for the intended laptop. It reports
`pass`, `fail`, `skip`, and `manual` statuses; skipped optional checks do not
fail the command unless `--require-real-model` is used. It never downloads
models, installs packages, changes system settings, or starts persistent
services. Voice push-to-talk remains a manual check because it needs local
microphone permission, configured whisper.cpp/Piper assets, and user-observable
audio I/O.

Multi-model Mac readiness verifies every configured local GGUF model that is
present and readable:

```bash
run april verify --all-configured-models \
  --require-real-model \
  --report data/verification/mac-readiness.json
```

`--mac-readiness` is an alias for `--all-configured-models`. Missing optional
specialist models are reported as skipped/degraded, never passed.
`--require-real-model` fails if no real configured GGUF model is exercised. A
fake/simulated runtime is never marked `real_model_verified`.

The Brain model must load, chat, stream, unload, return structured Brain JSON,
run routing evals, and meet `--min-routing-accuracy` (default `0.90`). Specialist
models must load, chat, stream, pass their role smoke check, and unload. Coding
and system-action smoke checks validate tiny JSON schemas. Prompts and generated
outputs are not stored in the report; only `smoke_kind`, `smoke_success`, and
`smoke_schema_valid` are recorded. Optional performance thresholds include
`--max-rss-mb`, `--min-tokens-per-second`, `--max-load-seconds`, and
`--max-first-token-latency-seconds`.

Strict Brain JSON routing must use real chat/structured backend support. If
llama.cpp falls back to prompt completion for a requested JSON response, Runtime
sets `diagnostics.structured_output_fallback: true`; real-model verification
treats that as a blocker rather than a valid strict-routing pass.

Multi-model reports keep the compatibility field `real_model_verified` ("at
least one real model passed") and add clearer levels:

- `none`: no real model was exercised and passed, or the backend is fake.
- `partial`: at least one real model passed, but the core set is not verified.
- `core`: brain passed, coding passed if configured, reading passed if
  configured, and the backend is real.
- `all`: every configured model exists, was exercised, passed acceptance gates,
  and specialist switching passed when applicable.

Single-model target-Mac verification remains available:

```bash
run april verify /absolute/path/to/model.gguf \
  --target-mac \
  --require-real-model \
  --report data/verification/single-model.json
```

Reports are redacted: no prompts, generated text, tokens, secrets, raw tool
arguments, file contents, or full paths. Model paths are basenames only. Real
verification requires local GGUF files and `llama-cpp-python`; APRIL never
downloads models or installs packages.

`real_model`, `voice_live`, and `workflow` reports are separate axes:

- `real_model` latest status includes only `multi_model` and `target_mac`
  reports.
- `voice_live` latest status includes only `voice_live` reports.
- `workflow` reports show local workflow coverage and do not imply real-model
  verification unless their sanitized payload explicitly says so.

`data/verification/` is generated and ignored by Git. The Core API exposes only
authenticated sanitized summaries through `GET /verification/report/latest`,
`GET /verification/report/latest?type=any`,
`GET /verification/report/latest?type=real_model`,
`GET /verification/report/latest?type=voice_live`,
`GET /verification/report/latest?type=workflow`,
`GET /verification/reports`, and
`GET /verification/reports/{report_basename}`. Report history is sorted by safe
report time (`generated_at`, then `timestamp`, then mtime fallback). The
basename endpoint rejects traversal, slashes, backslashes, symlinks, absolute
paths, non-JSON files, and arbitrary query paths. Desktop Readiness uses those
endpoints for separate real-model, workflow, voice-live, latest-report, and
report-history display.

## Semantic Memory Readiness

`run april memory doctor --json` is the offline readiness check for vector
memory. It reports the configured embedding provider, active vector-index
provider, dimensions, whether runtime-local was requested, whether APRIL is
falling back to hashed-token, whether reindex is required, whether an
embedding-role model is registered, whether that model path exists, the active
and effective generation, recovery state, and the last successful full reindex.
It does not start Runtime or load a model unless
`--verify-runtime-embedding` is passed, and that flag only probes
`/runtime/embed`. Runtime health also reports the typed batch capability and
its 64-item bound.

Use `run april memory repair-index` to inspect a malformed/missing/corrupt
`CURRENT` pointer without mutation. If it reports a validated recovery
candidate, apply exactly that repair with
`run april memory repair-index --apply`. If it reports no valid generation, run
`run april memory reindex`. Recovery mode is degraded readiness even though
SQLite memory remains available. Real semantic memory requires a runtime-local
embedding-role model and a reindex after switching providers.

Manual model guidance (APRIL does not download, register, or activate these
automatically):

```bash
run april model import --role embedding --id nomic-embed-text-v1.5 \
  --name "nomic-embed-text-v1.5 Q8" \
  --path /ABSOLUTE/LOCAL/PATH \
  --sha256 EXPECTED_SHA256
export APRIL_MEMORY_EMBEDDING_PROVIDER=runtime-local
export APRIL_MEMORY_EMBEDDING_MODEL_ID=nomic-embed-text-v1.5
run april memory doctor --verify-runtime-embedding
run april memory reindex --wait

run april model import --role reasoning --id qwen3-4b-reasoning \
  --name "Qwen3-4B Q4_K_M" --path /ABSOLUTE/LOCAL/PATH \
  --sha256 EXPECTED_SHA256
```

Deep and Council reasoning use the reasoning-role model when available and
otherwise report their Brain fallback honestly. Qwen3-4B Q4_K_M must be
benchmarked on the Intel MacBook before becoming a recommended default.

Fake soak is non-destructive and fake-backend-only:

```bash
run april verify --soak --fake --minutes 10 --report data/verification/soak.json
```

It repeatedly checks health, chat, and model listing with bounded delay, tracks
failures/latency/RSS when available, and never requires real models or voice.

Live voice verification is explicit and interactive:

```bash
run april voice verify-live --report data/verification/voice-live.json
```

It runs voice doctor, shows macOS microphone guidance, asks before recording,
uses push-to-talk only, runs local whisper.cpp and Piper if configured, stores
transcript length rather than transcript text, deletes temporary audio by
default, and never starts wake-word listening or uploads audio.

The conversation-live command is a separate, stricter real-hardware gate. It
guides two turns through wake detection, calibrated automatic endpointing,
whisper.cpp, the authenticated loopback Core API, Piper, playback, production
barge-in, and follow-up/session continuity. A natural 300–500 ms pause should
remain inside the first utterance; the default endpoint is 650 ms continuous
silence. Its report is redacted and only real-hardware evidence can set the
production verification flag. Existing one-turn reports are not upgraded.

`run april setup voice` never enables voice unless both `--apply --enable` are
present. `run april setup voice ... --apply` without `--enable` leaves
`voice.enabled: false`, even if it was previously true. A missing wake-word model
does not block push-to-talk, but wake-word listening remains unavailable or
unverified until a local wake-word model is configured and live verification
passes.

External actions such as git push, deployment, email, payment, and publishing
remain out of scope and disabled; they must not be simulated as successful.

Conversation-context verification uses the fake local Runtime client and
temporary SQLite databases. It covers incremental summary calls, complete-turn
checkpoints, stale compare-and-swap rejection, safe Reading-model degradation,
secret/raw-output omission, complete structured tool sequences, independent Core
category pre-bounds, and Runtime group diagnostics. These tests do not load or
claim verification of a real Reading GGUF.

The fake brain eval uses the deterministic fallback router and validates schema
validity plus routing expectations for ordinary chat, planning, coding,
reading, creative, reasoning, memory search/write, Git reads, patch proposals,
code edits, command execution, destructive/external requests, prompt injection,
path escape, secrets, unsupported tools, and malformed-routing recovery
coverage. Real-model evals run only with an explicit local GGUF path.
