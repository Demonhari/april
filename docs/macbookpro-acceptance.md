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

GGUF files are large local artifacts, so APRIL does not commit them and CI never
downloads them. You can either explicitly download APRIL's default core models
from the checked-in manifest or place existing local GGUFs yourself. Downloading
is not verification; real-model readiness remains false until load/chat/stream/
unload verification actually passes.

### Target Mac model install

```bash
run april model download --all-core --apply --yes
run april model doctor
run april setup mac-activation \
  --brain models/granite3.3-2b-q4_k_m.gguf \
  --coding models/qwen3-1.7b-q8_0.gguf \
  --reading models/qwen3-0.6b-q8_0.gguf \
  --skip-voice \
  --apply \
  --run-acceptance \
  --start-services
```

`run april model download` reads only `configs/model_downloads.yaml`, is dry-run
by default, and requires `--apply --yes` before network access starts. It writes
to `.part`, validates GGUF magic/size, atomically renames on success, records a
SHA-256, and registers the model through the same validated setup path. Use
`--skip-existing` to reuse existing targets or `--force` to overwrite them.
Reasoning remains optional and is not downloaded by default.

If you already have GGUF files, validate first with `--dry-run`, then apply. Use
**absolute paths**; APRIL never downloads models from model import/setup.

```bash
# Optional: add --reasoning /absolute/path/reasoning.gguf when configured.
run april setup models \
  --brain /absolute/path/brain.gguf \
  --coding /absolute/path/coding.gguf \
  --reading /absolute/path/reading.gguf \
  --dry-run

# Optional: add --reasoning /absolute/path/reasoning.gguf when configured.
run april setup models \
  --brain /absolute/path/brain.gguf \
  --coding /absolute/path/coding.gguf \
  --reading /absolute/path/reading.gguf \
  --apply

run april model doctor          # confirm each role resolves to a present file
run april model profile list    # inspect available hardware profiles
run april model profile apply intel_macbook_cpu_low   # or apple_silicon_macbook
```

Full Mac activation requires the core GGUF roles `brain`, `coding`, and
`reading`. `reasoning` is optional; configure it only when you have a separate
local model for deeper reasoning. If `--copy-into-models` is used and a later
role fails, APRIL restores `configs/models.yaml` and removes only files copied by
that failed command.

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
config only with `--apply`, and can chain straight into acceptance. It is
**dry-run by default** and never downloads models, installs packages, uses
`sudo`/Homebrew, or records audio. `--reasoning` / `--reasoning-id` are optional;
when omitted, the reasoning agent keeps using the configured brain model.

The wizard distinguishes partial model registration from full activation. Full
activation requires `brain`, `coding`, and `reading`, either supplied in the
command or already configured to existing local GGUF files. Supplying only part
of that core set fails before config writes by default. Use
`--allow-partial-model-set` only when you intentionally want to register the
supplied subset; the report is `incomplete`, includes
`core_model_set_complete: false`, lists `missing_required_roles`, and blocks
`--run-acceptance` until the core set is complete. Real GGUF verification still
requires an actual load/chat/stream/unload pass, and live voice verification
requires the live path to run. Fake verification remains explicitly
fake/simulated.

### Transactional apply and rollback

Apply is **transactional and validate-first**:

- Every supplied model and voice path is validated *before* anything is written.
  If validation fails, **nothing** is written.
- The model (`configs/models.yaml`) and voice (`configs/april.yaml`) config files
  are snapshotted, then applied in order.
- If a later step fails, the previous config is **restored automatically** so you
  never end up with models applied but voice half-written. The report's
  `transaction` block records `backup_created`, `committed`, `rolled_back`, and a
  redacted `rollback_reason`. (`--no-rollback` exists for debugging only and
  leaves the partial state in place; the default is always rollback-on-failure.)

### Enabling voice

The wizard configures voice paths but, by default, leaves voice **OFF** (no
surprises). Pass `--enable-voice` to turn voice on — but only after every required
voice artifact validates. `--enable-voice` may not be combined with `--skip-voice`.

```bash
# Models only — validate, apply, then run real-model acceptance and write a report:
# Optional: add --reasoning /absolute/path/reasoning.gguf when configured.
run april setup mac-activation \
  --brain /absolute/path/brain.gguf \
  --coding /absolute/path/coding.gguf \
  --reading /absolute/path/reading.gguf \
  --skip-voice \
  --apply \
  --run-acceptance \
  --write-report

# Full activation — models + voice enabled + full acceptance with live voice/wake
# and service orchestration:
# Optional: add --reasoning /absolute/path/reasoning.gguf when configured.
run april setup mac-activation \
  --brain /absolute/path/brain.gguf \
  --coding /absolute/path/coding.gguf \
  --reading /absolute/path/reading.gguf \
  --whisper-binary /absolute/path/whisper \
  --whisper-model /absolute/path/whisper-model.bin \
  --piper-binary /absolute/path/piper \
  --piper-model /absolute/path/piper-voice.onnx \
  --wake-word-model /absolute/path/april.onnx \
  --enable-voice \
  --apply \
  --run-acceptance \
  --acceptance-voice-live \
  --acceptance-wake-word-live \
  --start-services \
  --write-report
```

`--run-acceptance` runs **real-model** acceptance after a successful apply.
`--acceptance-voice-live` / `--acceptance-wake-word-live` add the live voice and
wake-word checks (both require `--run-acceptance` and `--enable-voice`, and are
incompatible with `--skip-voice`). `--start-services` orchestrates APRIL services
for those live checks using the same logic as `run april acceptance`; services the
wizard started are always stopped afterward unless `--keep-services-running`.
`--fake-services` cannot be combined with real-model acceptance. With
`--write-report`, a redacted report is written to
`data/verification/mac-activation-<timestamp>.json` (Git-ignored).

## Browsing reports

`run april reports` is a read-only browser over the redacted JSON reports under
`data/verification` (acceptance, mac-activation, voice-live, wake-word-live,
multi-model, fake-soak). It never prints tokens, transcripts, generated text, or
absolute paths.

```bash
run april reports list                          # newest first
run april reports latest                        # newest report of any known type
run april reports show data/verification/acceptance-….json
run april reports show-latest --type acceptance
run april reports show-latest --type mac_activation
run april reports clean --older-than-days 14 --dry-run
run april reports clean --older-than-days 14 --apply
```

`reports clean` is **dry-run by default** and deletes only `*.json` files older
than the given age that live directly inside `data/verification`; nothing outside
that directory is ever touched. The Core API exposes the same data read-only at
`GET /reports`, `GET /reports/latest`, and `GET /reports/latest/{report_type}`,
and the Desktop dashboard shows the latest acceptance/activation status.

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
