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
run april model doctor
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
models must load, chat, stream, pass their role smoke check, and unload. Optional
performance thresholds include `--max-rss-mb`, `--min-tokens-per-second`,
`--max-load-seconds`, and `--max-first-token-latency-seconds`.

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

The fake brain eval uses the deterministic fallback router and validates schema
validity plus routing expectations for ordinary chat, planning, coding,
reading, creative, reasoning, memory search/write, Git reads, patch proposals,
code edits, command execution, destructive/external requests, prompt injection,
path escape, secrets, unsupported tools, and malformed-routing recovery
coverage. Real-model evals run only with an explicit local GGUF path.
