# MacBook Pro Acceptance

This is the operator checklist for taking APRIL from a fresh clone to a
**verified real-model, real-voice** install on a MacBook Pro. It is built around
one gate command — `run april acceptance` — plus the exact setup and live-voice
commands that gate depends on.

Nothing here downloads a model, installs a Homebrew package, runs `sudo`, or
reaches the network. Every command is something *you* run; APRIL only validates,
reports, and (when you pass `--apply`) edits local config.

## What fake verification proves

`run april verify --fake` (and the fake-backend portion of `run april acceptance`)
runs the **deterministic launcher/workflow checks against a simulated runtime**.
A green fake run proves that:

- April Runtime and the Core API start, become healthy, and stop cleanly on
  loopback only.
- Routing, agents, tool dispatch, the deterministic permission engine, patch
  approval (immutable artifact bytes), path-escape/repo-override rejection, and
  audit/tool-call/agent-run records all behave correctly.
- Memory, conversations, projects, reminders, and the desktop SPA wiring work.
- Configuration (`configs/agents.yaml`, `configs/tools.yaml`,
  `configs/permissions.yaml`, `configs/models.yaml`) loads and validates.
- The full request → decision → tool-request → approval → audit path is intact.

In short: **the whole APRIL machine is correct**, end to end, with a stand-in
model.

## What fake verification does NOT prove

A green fake run says nothing about whether *real local inference* works on this
Mac. It does **not** prove:

- That `llama-cpp-python` is installed or that your GGUF files exist, load, and
  produce tokens.
- That the brain emits valid structured JSON, that routing accuracy holds on a
  real model, or that specialists load/unload while the brain stays resident.
- That tokens/sec, load time, first-token latency, or RSS are acceptable.
- That whisper.cpp, Piper, the microphone, the speaker, or the **wake word**
  actually work.

Fake mode uses a simulated backend, so it can *never* set `real_model_verified`
or `voice_live_verified`. To prove those, run the real-model and live-voice
commands below. `run april acceptance` reports a **warning** (not a pass) when
only fake verification succeeded and real models were not required.

## First-run bootstrap

```bash
run april setup bootstrap
run april setup tokens   # only if bootstrap reports token warnings
run april config validate
run april readiness
```

`setup bootstrap` ensures local directories, generates loopback tokens (values
are never printed), snapshots the machine, and recommends a model profile.
`readiness` then explains — offline and redacted — exactly what is still missing
for real-model and real-voice readiness.

## Configure GGUF models

Validate first with `--dry-run`, then apply. Use **absolute paths**; APRIL never
downloads models and never commits them.

```bash
run april setup models \
  --brain /absolute/path/brain.gguf \
  --coding /absolute/path/coding.gguf \
  --reading /absolute/path/reading.gguf \
  --dry-run

run april setup models \
  --brain /absolute/path/brain.gguf \
  --coding /absolute/path/coding.gguf \
  --reading /absolute/path/reading.gguf \
  --apply

run april model doctor          # confirm each role resolves to a present file
run april model profile list    # inspect available hardware profiles
run april model profile apply intel_macbook_cpu_low   # or apple_silicon_macbook
```

## Switch the runtime to llama_cpp

`configs/april.yaml` already defaults `runtime.backend: llama_cpp`. The fake
backend is only active when an environment override forces it. To run real
inference, install the runtime extra and make sure nothing pins the fake backend:

```bash
pip install -e '.[runtime]'          # installs llama-cpp-python (no Homebrew, no sudo)
unset APRIL_RUNTIME_BACKEND          # remove any APRIL_RUNTIME_BACKEND=fake override
# or, to be explicit:
export APRIL_RUNTIME_BACKEND=llama_cpp
run april readiness                  # 'runtime backend' and 'llama-cpp-python' should be ok
```

## Verify all configured models

```bash
run april verify --all-configured-models --require-real-model \
  --report data/verification/mac-readiness.json
```

This loads, chats, streams, and unloads **every** configured chat GGUF in one
runtime, verifies specialist switching keeps the brain resident, checks
structured brain JSON and routing accuracy, and fails if any configured chat
model is missing, skipped, unavailable, or fake. The report under
`data/verification/` is redacted (basenames, counts, booleans only) and is
Git-ignored.

## Configure whisper.cpp, Piper, and the wake word

Voice stays **OFF** until you explicitly enable it. Validate paths first, then
apply and enable:

```bash
run april setup voice \
  --whisper-binary /absolute/path/whisper.cpp/main \
  --whisper-model  /absolute/path/ggml-base.en.bin \
  --piper-binary   /absolute/path/piper \
  --piper-model    /absolute/path/voice.onnx \
  --wake-word-model /absolute/path/april.onnx \
  --dry-run

run april setup voice \
  --whisper-binary /absolute/path/whisper.cpp/main \
  --whisper-model  /absolute/path/ggml-base.en.bin \
  --piper-binary   /absolute/path/piper \
  --piper-model    /absolute/path/voice.onnx \
  --wake-word-model /absolute/path/april.onnx \
  --apply --enable

run april voice doctor    # microphone permission, devices, and configured paths
```

The wake-word model is a custom local openWakeWord model for "April"; APRIL never
downloads or trains one. Push-to-talk works without any wake-word model.

## Push-to-talk live verification

```bash
run april voice verify-live --report data/verification/voice-live.json
```

Records a short push-to-talk sample, transcribes it with whisper.cpp, synthesizes
a phrase with Piper, and asks you to confirm playback. The report stores only
transcript **length**, never transcript text. Temporary audio is deleted unless
you pass `--retain-debug-audio`.

## Wake-word live verification

Start APRIL services first so the Core `/voice/input` endpoint is reachable, then
run the wake-word gate in another terminal:

```bash
# terminal 1 — services (use --fake to verify the wake-word plumbing without
# real models; drop --fake once real GGUF models are configured):
run april --fake

# terminal 2 — the live wake-word check:
run april voice verify-wake-live --report data/verification/wake-live.json
```

It runs voice doctor, requires a configured wake-word model, confirms before
opening the microphone, then asks you to say **"April"** followed by a short
command. It verifies that microphone frames read, the wake word is detected, the
utterance is captured, whisper.cpp returns non-empty text, the wake word is
normalized out of the transcript, `/voice/input` answers, Piper synthesizes the
reply, and you confirm playback. Tunables: `--wake-wait-seconds`,
`--utterance-max-seconds`, `--retain-debug-audio`. The microphone is always
released on success, failure, timeout, cancellation, or Ctrl-C, and temporary
audio is deleted unless `--retain-debug-audio` is set.

## The acceptance gate

One command folds configuration validation, offline readiness, deterministic fake
verification, and (on request) real-model + live-voice checks into a single
`pass` / `warning` / `fail` status with copy-pasteable next actions. The report
records an honest **`acceptance_level`**:

- `fake_sanity` — only fake plumbing was proven.
- `real_models` — every configured GGUF model loaded/chatted/streamed/unloaded.
- `real_models_plus_voice` — real models **and** push-to-talk voice passed.
- `full_wake_voice` — real models, push-to-talk voice, **and** wake word passed.

`run april acceptance` is **fake/local sanity only**. It reports a **warning**
(exit 0), not a pass, unless you require real models — or explicitly opt in with
`--allow-sanity-pass` for a clean fake-only run. A fake run can never look like
full Mac readiness, and `--require-real-models` **fails** if the runtime backend
is fake or any configured chat model is missing/unavailable.

```bash
# Fake/local sanity (warning unless --allow-sanity-pass):
run april acceptance --write-report

# Real-model acceptance:
run april acceptance --require-real-models --write-report

# Wake-word plumbing with fake services (no real models, no real backend):
run april acceptance --wake-word-live --start-services --fake-services --write-report

# Full target-Mac acceptance (real models + both live-voice paths):
run april acceptance \
  --require-real-models \
  --voice-live \
  --wake-word-live \
  --start-services \
  --write-report
```

### Service orchestration

Live voice and wake-word checks call the Core `/voice/input` endpoint, so the API
must be running. `--start-services` lets acceptance become a true one-command gate:

- `--start-services` starts any missing APRIL services before the live checks.
- `--fake-services` starts them with the **fake** runtime (plumbing only). It may
  not be combined with `--require-real-models` (a fake runtime cannot verify real
  models) and requires `--start-services`.
- Services that acceptance started are **always** stopped at the end — including
  on failure, timeout, cancellation, or Ctrl-C — unless `--keep-services-running`.
- `--service-timeout FLOAT` bounds startup health-wait.

The report's `services` block records `mode`, `started_by_acceptance`,
`stopped_after_acceptance`, `startup_status`, `shutdown_status`, and API/runtime
reachability.

Useful options: `--report PATH` (write to a chosen path), `--json`,
`--max-output-tokens`, `--timeout`, and the optional performance gates
`--min-tokens-per-second`, `--max-load-seconds`,
`--max-first-token-latency-seconds`, `--max-rss-mb`. With `--write-report` and no
`--report`, the redacted report is written to
`data/verification/acceptance-<timestamp>.json` (Git-ignored). Reports never
contain tokens, transcripts, generated text, or absolute paths. A `warning` exits
0; only `fail` exits 1.

## The Mac activation wizard

`run april setup mac-activation` is one guided local command that validates the
intended GGUF model set and (unless `--skip-voice`) the local voice tools, writes
config only with `--apply`, and can chain straight into real-model acceptance. It
is **dry-run by default** and never downloads models, installs packages, uses
`sudo`/Homebrew, or records audio.

```bash
# Validate paths only (writes nothing):
run april setup mac-activation \
  --brain /absolute/path/brain.gguf \
  --coding /absolute/path/coding.gguf \
  --reading /absolute/path/reading.gguf \
  --dry-run

# Apply config, then run real-model acceptance and write a report:
run april setup mac-activation \
  --brain /absolute/path/brain.gguf \
  --coding /absolute/path/coding.gguf \
  --reading /absolute/path/reading.gguf \
  --apply \
  --run-acceptance \
  --write-report
```

Add `--whisper-binary/--whisper-model/--piper-binary/--piper-model`
(and optional `--wake-word-model`) to validate/apply voice paths too, or
`--skip-voice` to activate models only. The wizard configures voice paths but
never **enables** voice — turn it on explicitly later with
`run april setup voice ... --apply --enable`. With `--write-report`, a redacted
report is written to `data/verification/mac-activation-<timestamp>.json`
(Git-ignored).

## Troubleshooting

| Symptom | What it means | Fix |
| --- | --- | --- |
| `llama-cpp-python` missing | Readiness blocker "llama-cpp-python"; real verification reports the runtime extra is absent | `pip install -e '.[runtime]'` (no Homebrew, no sudo) |
| Missing GGUF path | Readiness blocker "configured GGUF model files"; a model role resolves to a non-existent file | `run april setup models --<role> /absolute/path/model.gguf --apply`, then `run april model doctor` |
| Fake backend still active | Readiness "runtime backend" blocker says backend is `fake`; reports show `runtime_backend: fake` and never set `real_model_verified` | `unset APRIL_RUNTIME_BACKEND` (or `export APRIL_RUNTIME_BACKEND=llama_cpp`); confirm `configs/april.yaml` has `runtime.backend: llama_cpp` |
| Default development tokens | Readiness "api/runtime tokens" warning: `local-dev-token` / `local-dev-runtime-token` still active | `run april setup tokens` (rotates loopback tokens; values are not printed) |
| Microphone permission denied | Voice doctor "microphone access: permission_or_device"; querying devices fails though sounddevice is installed | macOS: System Settings → Privacy & Security → Microphone → allow your terminal app, then re-run `run april voice doctor` |
| No input device | Voice doctor "microphone access: no_input_device" / "input devices: 0" | Connect a microphone and grant permission; `run april voice devices` should then list an input device |
| whisper.cpp failed | STT step errors; doctor shows the whisper binary/model path as missing or the run reports "whisper.cpp failed" | Re-check `--whisper-binary` / `--whisper-model` absolute paths via `run april setup voice ... --dry-run`; confirm the binary is executable |
| Piper failed | TTS step errors; doctor shows the Piper binary/model path as missing or the run reports "Piper failed" | Re-check `--piper-binary` / `--piper-model` absolute paths via `run april setup voice ... --dry-run`; confirm the binary is executable |
| Wake word not detected | `verify-wake-live` reports `wake_word_detected=false` and a "no wake word" skip | Confirm `voice.wake_word_model_path` points at a present openWakeWord "April" model; lower `voice.wake_word_threshold` if needed; speak "April" clearly, then the command. Push-to-talk (`run april voice verify-live`) needs no wake-word model |
