# APRIL Implementation Plan

## Milestones

1. Project foundation and April Runtime
   - Create Python project metadata, settings, configs, and test tooling.
   - Implement model registry validation, backend interface, fake backend, llama.cpp adapter boundary, model lifecycle, generation locking, runtime API, and streaming contract.
   - Add runtime tests that require no real model files.

2. Local security foundation
   - Implement SQLite migrations, memory repositories, audit logging, permission engine, one-time approvals, path security, read-only filesystem tools, Git read-only tools, and command policy.
   - Add security tests for path traversal, symlink escape, approval replay, command denial, and risky tool gating.

3. Brain, agents, core API, and CLI
   - Implement strict brain JSON parsing with one repair attempt and deterministic fallback routing.
   - Implement agent registry and simple bounded agent execution through April Runtime.
   - Implement core orchestrator, authenticated API, runtime model proxying, and Typer/Rich CLI.

4. Memory retrieval and repository indexing
   - Implement durable memory policy, SQLite memory operations, deterministic local hashed-token embeddings, vector index persistence, hybrid retrieval hooks, repository indexing, patch proposal boundary, and configured test runner.

5. Optional local model and voice adapters
   - Implement isolated `llama-cpp-python` backend adapter with graceful missing-dependency errors.
   - Implement optional voice health, push-to-talk pipeline, whisper.cpp subprocess STT adapter, Piper subprocess TTS adapter, and fake adapters for tests.

6. Documentation and quality gates
   - Complete README and architecture/security documents.
   - Provide safe scripts for setup, runtime, API, CLI, model placement help, and repository indexing.
   - Run tests, Ruff, mypy, and required source scans.

7. Proactive scheduler
   - Add a pure-asyncio poll loop that fires due reminders through a pluggable notification sink (log by default, optional native macOS banner).
   - Add daily briefings: a plain-text summary of open tasks, reminders due in the next 24 hours, and project count, composed without any LLM or external I/O.
   - Off by default: neither reminders nor briefings run unless `scheduler.enabled` / `scheduler.briefing_enabled` are set; the loop and briefings are never activated implicitly.
   - Restart-safe: the last briefing date is persisted in a `scheduler_state` table so a briefing fires at most once per local day even across process restarts.
   - Reminder and briefing paths are independent inside each tick; a failure in one is audited and never blocks the other. A `GET /scheduler/briefing/preview` endpoint and `run april briefing` command let the user view today's briefing on demand regardless of enabled state.

8. Desktop UI
   - Serve a local single-page UI from the Core API at `GET /desktop` using a `StaticFiles` mount of `apps/desktop/web/`. The SPA is plain static HTML/CSS/JS — no Node, npm, or build step — and reuses the existing authenticated endpoints; it adds no public surface and keeps `/health` the only unauthenticated, redacted route.
   - Add one authenticated, strictly allowlisted endpoint, `GET /diagnostics/activity?limit=N` (capped at 200), that projects the sanitized audit JSONL down to event type, timestamp, reference IDs, and risk level — never prompt content, file contents, tool arguments, tokens, or secrets.
   - Add a `run april desktop` launcher that ensures Runtime + Core API (honoring `--fake`), never starts voice/wake-word/microphone, resolves the API token from the same settings/.env source as the CLI, and opens `http://127.0.0.1:<api_port>/desktop#token=<TOKEN>` with the token in the URL fragment only. An optional native window behind the `[desktop]` extra (pywebview) injects the token via the JS bridge so it never appears in a URL; absent pywebview it falls back to the browser path.
   - The SPA holds the token in memory only (never `localStorage`/`sessionStorage`), strips the fragment via `history.replaceState` on load, streams Chat via `fetch()` + `ReadableStream` against `POST /chat/stream`, routes `approval_required` to the exact-ID Approvals screen (a chat "yes" is never approval), and surfaces 401/403/network errors in a non-crashing banner. Screens: Chat, Projects, Approvals, Memory, Reminders & Tasks (+ briefing), Status & Models, and Activity/Logs.
   - Add Readiness report history through authenticated sanitized endpoints
     (`/verification/report/latest`, `/verification/reports`, and basename
     lookup) that read only `data/verification/*.json` and reject traversal,
     symlinks, non-JSON files, and arbitrary path input.
   - Add discoverable local setup helpers for model paths, voice paths, and the
     unsigned app stub (`run april setup models`, `run april setup voice`,
     `run april setup app-stub`), all dry-run or explicit-apply where they
     mutate config.
   - Tests run on the fake backend with no GGUF/network/microphone: the static mount returns `index.html`, `/diagnostics/activity` requires auth and is redacted, report history is sanitized, and the `desktop` subcommand resolves config and target URL without launching a real browser.

9. Sentinel wake and sessions
   - Add the single-owner Sentinel microphone loop with local two-stage wake detection, ring-buffer STT confirmation, mute release, follow-up capture, earcons, and soft speaker-gate fallback.
   - Add the local Unix-socket wake bus, cross-surface session continuity, explicit session close/reflection, and persisted wake feedback without transcript leakage.

10. apriald and resource governor
   - Add the single-instance `apriald` supervisor for Runtime, Core API, and explicitly enabled Sentinel, with health status, bounded restart backoff, and user LaunchAgent support.
   - Add local RAM, CPU, power, and idle policy gates for resident/background work, model prewarming, and load-time generation thread budgets.

11. Archive memory evolution
   - Add session-close Archive reflection with strict local JSON, sensitivity and confidence filtering, duplicate consolidation, contradiction tracking, bounded machine-memory decay, and inspectable user-model guidance.
   - Add deterministic two-stage retrieval with an optional typed local Runtime reranker and audited fallback.

12. Playbooks
   - Add deterministic mining of repeated successful tool sequences into fenced playbook candidates, with safe-trigger checks and permission-derived adoption gates.
   - Add exact-action approval for Level 3+ adoption and bounded execution through the trusted tool context.

13. Dreamer
   - Add the off-by-default nightly D1-D6 replay, distill, mine, evolve, examine, and report pipeline behind scheduler, resource, kill-switch, time, and write-fence gates.
   - Add disarmed phase execution, deterministic reports, eval staging, prompt-overlay ratchets, exact-hash approval, versioning, and rollback.

14. Intelligence ladder
   - Add deterministic reflex, normal, deep, verified, and council rungs with confidence thresholds, whole-rung budgets, local model selection, and safe fallback.
   - Add fenced, ratcheted, versioned ladder-threshold overlays with rollback and stock restoration.

15. LoRA adapter lifecycle
   - Add fenced adapter evidence, hash-bound activation pointers, perplexity and production real-model verification gates, version history, and rollback.
   - Keep adapter training and quality evaluation operator-supervised; APRIL never downloads, trains, or silently activates an adapter.

## Current Status (honest)

This is the candid state of the MVP. A large, green automated suite verifies
orchestration, permissions, memory, and API/desktop contracts against the
deterministic fake backend — it does **not** verify real models, live audio, or
native packaging.

- **Core fake-backend MVP: implemented and tested.** Runtime, Brain routing,
  specialist agents, the permission engine with exact-action approvals, SQLite
  memory, the scheduler, documents, and the desktop SPA all pass against
  `APRIL_RUNTIME_BACKEND=fake`.
- **v2 control plane: implemented and fake-verified.** Sessions, explicit and
  wake feedback, Archive reflection, Dreamer overlays, versioned prompt and
  ladder-threshold rollback, gated LoRA adapter pointers, and named agent pool
  stats all write only to fenced evolution/playbook paths or allow-listed DB
  tables. Deleting `data/evolution/` restores stock prompt, ladder, and adapter
  behavior.
- **D4 learned guidance: implemented in two guarded tiers.** Always-on Tier A
  deterministically synthesizes advisory lines from recent Archive/Dreamer
  correction memories, surviving facts from adjudicated contradiction pairs,
  and negative-feedback reasons. It orders by recency/confidence, deduplicates,
  and attributes session-backed evidence through the originating agent run.
  Optional Tier B (`evolution.model_drafted_overlays: false` by default) lets
  Archive request one advisory draft from the same inputs through the typed
  local Runtime client. An unavailable Runtime is audited and skipped. Both
  tiers keep the two-candidate cap, character budget, structural rejection at
  generation/approval/load, D5 ratchet, and Forge/Hand approval gate.
- **Agent identity and prompt layering: implemented and fake-verified.** Every
  agent prompt starts with its stable call sign, mandate, and non-goals. The
  base prompt bytes remain first and immutable; learned guidance is appended
  after the mandate and cannot change agent config or policy.
- **Runtime adapter boundary: enforced by import test.** Pure adapter pointer
  readers and hashing live in `april_common.adapter_pointer` and are re-exported
  by `services.evolution.adapters`. April Runtime imports the leaf module only,
  keeping `services.evolution`, `services.memory`, and API SQLite code out of
  its process import graph while preserving config-path > pointer > none
  precedence and existing real/fake load behavior.
- **Cold fake verification: hardened without changing scores.** The launcher
  performs one unscored tool-routing warm-up, retries only the first scored
  tool-routing response once when `result` is missing, and includes a bounded
  response-body snippet if that failure persists. Unit tests exercise the
  helper logic without starting servers.
- **Real GGUF: not verified until you run the real-model checks.** The default
  backend is `llama_cpp`, but no GGUF is downloaded or committed. Run
  `run april readiness` to see exactly what is missing, then
  `run april verify --all-configured-models --require-real-model` to actually
  load/chat/stream/unload your local models. Until then, real-model readiness is
  `none`.
- **LoRA training and adapter quality evidence: guided but operator-supplied.**
  APRIL can validate and split a reviewed dataset, then launch only explicitly
  configured local trainer and evaluator executables through an exact-approved
  durable job. It never installs or downloads them, never invents scores, and
  registers successful output only as an inactive candidate. Activation still
  requires reviewed evidence and, in production, a fresh real-model
  verification report that loaded the same adapter hash.
- **Voice: code exists, live voice is unverified here.** Voice is off by default
  and requires the `.[voice]` extra plus whisper.cpp / Piper / wake-word
  binaries and models you install yourself, and macOS microphone permission.
  `run april voice doctor` distinguishes a *missing dependency* from a
  *permission/device failure* and reports push-to-talk fallback availability
  (push-to-talk needs no wake-word model). Live audio is verified only by
  `run april voice verify-live` on your Mac.
- **Wake-word and speaker gate: separate target-Mac blockers.** Sentinel has
  fake-tested wake routing, feedback verbs, generated earcon plumbing, and
  `idle | listening | muted` status. Live wake-word still requires a local
  openWakeWord ONNX and `run april voice verify-wake-live`. The speaker gate
  accepts `off | soft`; the shipped `OnnxSpeakerVerifier` scores bounded local
  PCM through an operator-configured raw-waveform embedding ONNX, silently drops
  non-matches with an audit event, and degrades to off with one audited warning
  when the model or optional runtime is unavailable. The manual setup runbook is
  `scripts/speaker_verifier/README.md`. No embedding model ships with APRIL, so
  supplying and validating that model remains a target-Mac blocker. This filter
  is convenience only, never authentication or a permission boundary.
- **Dreamer pauses between phases when the Mac becomes busy.** With
  `evolution.recheck_governor_between_phases` enabled, the same Resource Governor
  used at entry is consulted between D1-D5 work phases. A denial skips the
  remaining work phases with an audited reason while D6 still writes the report
  and morning briefing line.
- **Governor generation threads: implemented at model-load granularity.** The
  injected activity/idle policy selects 6 threads for an active or unknown user
  signal and 8 for a trusted idle signal. Core orchestration transports the
  budget through typed Runtime requests; Runtime alone applies it as
  `llama_cpp.Llama(n_threads=...)`, and the fake backend records the effective
  model definition. `llama-cpp-python` fixes `n_threads` at model construction,
  so a changed budget takes effect on the next safe model load/reload rather
  than changing an in-flight generation.
- **Desktop: local SPA / optional native wrapper plus explicit production packaging.** The UI
  is plain static HTML/CSS/JS served over authenticated loopback. The optional
  `dist/APRIL.app` stub is a development launcher only — no signing, no
  notarization, no bundled models/voice/tokens, ignored by Git. The separate
  `run april package` path builds and validates a production bundle and exposes
  operator-driven signing, notarization, stapling, Gatekeeper, and owner
  LaunchAgent commands without storing Apple credentials.
- **Memory vector search defaults to hashed-token embeddings.** Semantic
  `runtime-local` embeddings are used only when a local embedding-role GGUF is
  registered and `memory.embedding_provider=runtime-local` is set and verified.
  All index readers/writers resolve the same configured provider (shared
  `vector_memory_from_settings` factory) so vector spaces are never silently
  mixed; an unavailable embedding model falls back to hashed-token with an
  audited warning.
- **External actions remain out of scope and disabled.** No git push, deploy,
  email, payments, automatic model downloads, telemetry, cloud APIs, or broad
  delete. The only file-removing flow is the scoped, Level-4 approval-gated
  log/cache cleanup.
- **Remaining v2 operator blockers are explicit.** The production speaker
  verifier adapter and manual runbook ship, but no speaker-embedding model is
  present; the operator must supply and validate one on the target Mac. The
  guided LoRA job remains operator-supervised and requires locally supplied,
  reviewed trainer and evaluator executables.
  Governor load-time thread throttling and the dedicated SPA Adapters screen are
  implemented; neither claims target-Mac model quality or hardware validation.

## Architectural Assumptions

- The repository root is the default APRIL home unless `APRIL_HOME` points elsewhere.
- The core API and April Runtime are separate FastAPI processes communicating over loopback HTTP.
- Model files are referenced only by registered model IDs from `configs/models.yaml`.
- Tests and local development can use `APRIL_RUNTIME_BACKEND=fake`.
- Specialist agent loops are intentionally conservative in the MVP: tool execution is deterministic and bounded, while generated responses come through the runtime client.
- Deep reasoning (architecture mode) is functional: the Reasoning Agent defaults to the brain model and automatically upgrades to a registered `role: reasoning` model when the runtime reports one as available, failing safe to the brain model on any error. Council mode defaults to the same shared-model best-of-N behavior; optional `deep_mode.council_mode=multi_agent` uses reasoning/general/creative agent model IDs only when at least two distinct models resolve, otherwise it records a fallback to best-of-N.
- The default vector embedding is a deterministic hashed-token baseline, not a semantic local embedding model.

## Important Security Decisions

- The brain model cannot grant or lower permissions. Tool policy and argument-sensitive checks decide the authoritative permission level and risk.
- All Level 3+ actions create pending approvals and do not execute until a later exact-action approval is consumed.
- Filesystem tools resolve paths and nearest existing parents before access, block symlink escapes, reject null bytes, cap file sizes, and deny sensitive locations.
- Shell execution is restricted to configured argv allowlists. Model-controlled commands never enable shell execution.
- Audit records are append-only JSONL. Risky operations fail closed if approval or audit state fails.
- Voice is never activated by API startup and must be explicitly invoked.

## Known External Dependencies

- Base runtime: FastAPI, Uvicorn, Pydantic v2, pydantic-settings, PyYAML, HTTPX, aiosqlite, Typer, Rich, NumPy, and Jinja2.
- Development: pytest, pytest-asyncio, pytest-cov, Ruff, and mypy.
- Optional runtime: `llama-cpp-python`.
- Optional voice: `sounddevice`, `openwakeword`, local `whisper.cpp` binary, and local Piper binary/model.
- No Homebrew, model download, cloud model API, telemetry, or microphone/speaker access is required by tests.
