# Run APRIL Verification

APRIL release checks should include these local launcher gates:

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
run april memory doctor
run april eval brain --fake
```

## Exact Target Mac Order

Run target-Mac setup and real verification in this order:

1. `run april readiness`
2. `run april setup bootstrap`
3. `run april setup tokens` if bootstrap reports token warnings
4. `run april model profile apply intel_macbook_cpu_low`
5. `run april setup models --brain /absolute/path/brain.gguf --coding /absolute/path/coding.gguf --reading /absolute/path/reading.gguf --dry-run`, then repeat with `--apply`
6. `pip install -e '.[runtime]'`
7. `run april verify --all-configured-models --require-real-model --report data/verification/mac-readiness.json`
8. `run april verify --workflow --real-model --report data/verification/workflow-real.json`
9. Optional voice setup/doctor/live verification:
   `run april setup voice --whisper-binary /path/to/whisper.cpp/main --whisper-model /path/to/ggml-base.en.bin --piper-binary /path/to/piper --piper-model /path/to/voice.onnx --dry-run`,
   `run april voice doctor`,
   `run april voice verify-live --report data/verification/voice-live.json`

Blank API tokens never authenticate. If `APRIL_API_TOKEN` is empty, protected
Core API endpoints fail closed with an auth/config error; token values are not
printed in responses. The local development default `local-dev-token` remains
valid in development/test.

`run april setup bootstrap` warns on known development tokens, placeholder
tokens, blank API tokens, and blank/missing Runtime tokens without printing token
values. JSON output redacts local absolute paths by default; use `--show-paths`
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
run april model benchmark /absolute/path/to/small-local-model.gguf --runs 1 --max-output-tokens 32
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
embedding-role model is registered, and whether that model path exists. It does
not start Runtime or load a model unless `--verify-runtime-embedding` is passed,
and that flag only probes `/runtime/embed`. Real semantic memory requires a
runtime-local embedding-role model and `run april memory reindex` after switching
providers.

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

`run april setup voice` never enables voice unless both `--apply --enable` are
present. `run april setup voice ... --apply` without `--enable` leaves
`voice.enabled: false`, even if it was previously true. A missing wake-word model
does not block push-to-talk, but wake-word listening remains unavailable or
unverified until a local wake-word model is configured and live verification
passes.

External actions such as git push, deployment, email, payment, and publishing
remain out of scope and disabled; they must not be simulated as successful.

The fake brain eval uses the deterministic fallback router and validates schema
validity plus routing expectations for ordinary chat, planning, coding,
reading, creative, reasoning, memory search/write, Git reads, patch proposals,
code edits, command execution, destructive/external requests, prompt injection,
path escape, secrets, unsupported tools, and malformed-routing recovery
coverage. Real-model evals run only with an explicit local GGUF path.
