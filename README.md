# APRIL

APRIL is a private, local-first AI assistant MVP for macOS. It is CLI-first, uses a separate local model service called April Runtime, supports specialist agents, stores inspectable local memory, and enforces deterministic tool permissions with exact-action approvals.

No model files are downloaded automatically. No cloud AI APIs, Ollama integration, telemetry, or unrestricted shell execution are included.

## Architecture

```mermaid
flowchart TD
  UI[CLI / future desktop] --> API[Core APRIL API<br/>127.0.0.1:8765]
  API --> Brain[April Brain Router]
  Brain --> Agents[Specialist Agents]
  Agents --> Tools[Typed Tool Registry]
  Agents --> Runtime[April Runtime<br/>127.0.0.1:8766]
  Runtime --> Llama[llama-cpp-python]
  Llama --> Model[Local GGUF model]
  API --> Memory[(SQLite + local vector index)]
```

Only `services/april_runtime/llama_cpp_backend.py` imports `llama_cpp`. Agents and the core API talk to models through HTTP requests to April Runtime.

## Install

APRIL supports Python 3.11 through 3.13 for the Core MVP. Optional local
runtime and voice dependencies remain adapter-isolated.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Or:

```bash
make install-dev
```

## Configuration

Defaults live in `configs/april.yaml`, `configs/models.yaml`,
`configs/agents.yaml`, `configs/tools.yaml`, and `configs/permissions.yaml`.
These files are active runtime policy, not documentation-only examples.
Environment overrides use the `APRIL_` prefix for local machine settings such
as ports, data paths, model backend, and allowed roots.

Useful local development settings:

```bash
export APRIL_RUNTIME_BACKEND=fake
export APRIL_API_TOKEN=local-dev-token
export APRIL_ALLOWED_FILESYSTEM_ROOTS="$PWD"
```

Both APIs bind to `127.0.0.1` by default. CORS is disabled by default.
The example tokens are development-only. For a non-development local setup,
generate random local tokens without printing them:

```bash
run april setup tokens --output .env
```

Set `APRIL_ENV=production` only after replacing the example tokens. In
production mode APRIL rejects known development tokens at startup.

Development installs can use direct dependency constraints without pulling in
optional runtime or voice wheels:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev]' -c constraints-dev.txt
```

## Local Models

APRIL never downloads models. Register existing local GGUF files with:

```bash
run april model import --role brain --id april-brain --name granite3.3-2b --path /absolute/path/model.gguf
run april model import --role coding --id april-coding --name qwen3-1.7b --path /absolute/path/model.gguf
run april model import --role reading --id april-reading --name qwen3-0.6b --path /absolute/path/model.gguf
```

If the source file is outside configured allowed roots, copy it into APRIL with:

```bash
run april model import --role brain --id april-brain --name granite3.3-2b --path /absolute/path/model.gguf --copy-into-models
```

CPU-only profiles are local config edits only:

```bash
run april model profile list
run april model profile apply intel_macbook_cpu_low
run april model doctor
run april verify --real-model /absolute/path/model.gguf
run april model benchmark /absolute/path/model.gguf --runs 1 --max-output-tokens 32
```

Missing files do not crash startup. Runtime health reports degraded status. Use `APRIL_RUNTIME_BACKEND=fake` for tests and development without model files.

## Hari's Local Setup Path

1. Install APRIL:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev]'
make install-global-force
export PATH="$HOME/.local/bin:$PATH"
```

2. Import local GGUF models with `run april model import`.
3. Apply the Intel MacBook CPU profile:

```bash
run april model profile apply intel_macbook_cpu_low
```

4. Run model and fake verification:

```bash
run april model doctor
run april verify --fake
run april verify --real-model /absolute/path/model.gguf
run april eval brain --real-model /absolute/path/model.gguf
```

5. Configure local voice paths in `configs/april.yaml`, then run:

```bash
run april voice doctor
run april voice test-record --seconds 3
run april voice test-stt /path/to/audio.wav
run april voice test-tts "Hello Hari"
```

6. Start daily CLI usage:

```bash
run april --fake --oneshot ask "April, plan my work today."
run april ask "April, plan my work today."
```

## Start Services

Terminal 1:

```bash
make run-runtime
```

Terminal 2:

```bash
make run-api
```

CLI:

```bash
make cli
april health
april ask "April, plan my work today."
april models
```

## Run APRIL From Any Folder

Recommended zsh setup:

```bash
cd april
scripts/setup_mac.sh --base --global --add-to-path
source ~/.zshrc
run april --fake
```

Alternative without modifying shell config:

```bash
cd april
scripts/setup_mac.sh --base
make install-global
export PATH="$HOME/.local/bin:$PATH"
run april --fake
```

Fallback that always works after install, even before PATH reload:

```bash
"$HOME/.local/bin/run" april --fake
```

`run april` locates `APRIL_HOME`, starts April Runtime and the Core API when
they are missing, waits for both localhost health checks, then opens interactive
CLI chat. It does not start voice, wake-word, or microphone services. Services
still bind to `127.0.0.1`. Real GGUF models are optional for MVP testing; use
`--fake` to run with the fake backend.

Useful launcher commands:

```bash
run april --fake
april-run doctor
run april doctor
run april config validate
run april verify --fake
run april verify --workflow
run april verify --target-mac
run april model doctor
run april model profile list
run april status --json
run april status
run april stop
run april restart
run april logs --tail 100
run april ask "April, plan my work today."
run april health
run april models
run april approvals
run april approve APPROVAL_ID
run april deny APPROVAL_ID
run april agent run coding_agent "Inspect this repository" --project-id PROJECT_ID
run april config inspect
run april setup tokens --output .env
run april reminder list
run april reminder create "stand up" --due-at 2026-06-21T09:00:00Z
run april reminder delete REMINDER_ID
run april task list
run april briefing
run april voice health
run april voice doctor
run april voice devices
run april voice test-record --seconds 3
run april voice test-stt /path/to/audio.wav
run april voice test-tts "Hello Hari"
run april voice ptt
run april voice listen
run april memory doctor
run april eval brain --fake
```

`run april --fake` starts missing services with `APRIL_RUNTIME_BACKEND=fake`
without editing `.env`. Services still bind to `127.0.0.1`; PID files are under
`data/run/`, and logs are written to `logs/runtime.log` and `logs/api.log`.

Uninstall only APRIL-owned wrappers:

```bash
make uninstall-global
```

Troubleshooting `zsh: command not found: run`:

Cause: the APRIL wrapper is not installed or `~/.local/bin` is not in PATH.

Temporary fix for the current shell:

```bash
cd april
make install-global
export PATH="$HOME/.local/bin:$PATH"
run april --fake
```

Permanent zsh fix:

```bash
cd april
make install-global-path
source ~/.zshrc
run april --fake
```

If `run` resolves to a different command, inspect with:

```bash
april-run doctor
```

Then force-replace only when you intend to replace the existing `run` command:

```bash
make install-global-force
```

## Approval Example

```bash
april ask "Apply the fix." --project-id PROJECT_ID
april approvals
april approve APPROVAL_ID
```

APRIL never treats a casual "yes" inside chat as approval. Approval must reference the exact approval ID or use the dedicated CLI/API approval flow. Before an approved tool runs, APRIL reloads the approval, revalidates current tool policy for the scoped agent, verifies the exact argument hash, records the tool call, consumes the approval once, and audits the outcome.

For natural chat code changes such as `april ask "Apply the fix." --project-id
PROJECT_ID`, the Coding Agent now runs through the structured specialist loop:
it can inspect project files, ask `patch_generator` to create an immutable patch
artifact, ask `patch_applier` to apply it, suspend for approval, and resume the
same agent run after approval returns the exact tool result.

Patch proposals are stored in APRIL's content-addressed artifact store under
`data/artifacts/patches/`. The artifact may live outside the selected
repository, but every patch target must still resolve inside the selected
project. Patch approvals bind the artifact ID, patch SHA-256, exact byte length,
affected paths, selected project ID, repository root, available Git state,
expected side effects, expiry, and approval ID. Before applying, APRIL loads the
approved immutable bytes, recalculates the digest, validates target paths again,
then runs `git -C REPO apply --check -` and `git -C REPO apply -` against those
same in-memory bytes. Git commit approvals bind the exact staged diff digest,
staged tree ID, commit message, and repository identity.

## Repository Analysis Example

```bash
export APRIL_ALLOWED_FILESYSTEM_ROOTS="$PWD"
april project add "$PWD"
april ask "April, check why the animation in this repository is broken." --project-id PROJECT_ID
```

Repository work requires an explicit selected project through `project_id` or `repo_path`; APRIL no longer guesses a repository from the first allowed root. The coding agent can use read-only Git and filesystem tools without approval. File edits, patch application, test execution, and commits require approval.

When a project is selected, APRIL derives project-scoped tool roots from trusted
application state. Model-provided repository roots or absolute file paths cannot
override the selected project.

## Streaming

`POST /chat/stream` uses real runtime streaming. The Core API routes the request, runs permitted tools, stops immediately for approvals, and then forwards token events from April Runtime without buffering the full response. SSE events include `meta`, `token`, `approval_required`, `usage`, `done`, and `error`.

## Conversations

`POST /chat` accepts an optional `conversation_id`. If omitted, APRIL creates a
local conversation and returns its ID in `result.conversation_id`. The
interactive CLI creates one conversation ID per chat session and reuses it for
every turn. Recent bounded history is included in the next agent prompt as
context, not instructions.

Conversations are bound to either a selected project ID or explicit no-project
scope. APRIL rejects attempts to reuse a project conversation with a different
project. Brain routing and code-modification planning receive bounded recent
history as context.

## Structured Agents

Specialist agents use the structured loop by default for `/chat` and
`/agents/run`: Coding, Reading, Reasoning, System Action, and Creative when it
requests tools. General Agent simple chat remains a direct model response.

Specialist output must be exactly one JSON object: `final_answer`,
`tool_request`, `approval_required`, or `structured_error`. The loop enforces the
configured agent model, allowed/blocked tools, maximum iterations, and
permission gates. Level 3+ tool requests create exact approvals and persist a
suspended run. Approving the ID executes the exact tool once, appends the
sanitized result, and resumes the same run. `APRIL_LEGACY_ORCHESTRATOR=1`
temporarily restores the previous planned-tool path for compatibility testing.

### Deep Reasoning

Deep reasoning ("architecture mode") is always available. The Reasoning Agent
runs on the brain model by default, so requests like "reason through the
trade-offs", "compare approaches", or "weigh the options on this architectural
decision" route to it and return a real answer with no extra setup.

If you register a larger model with `role: reasoning`, APRIL automatically uses
it for reasoning runs whenever the runtime reports it as available, and falls
back to the brain model on any error. Register one with:

```bash
run april model import --role reasoning --id april-reasoning \
  --path models/your-reasoning-model-q4_k_m.gguf
```

See the commented `reasoning:` example in `configs/models.yaml`. The Reasoning
Agent is read-only (it keeps `read_file`, `search_files`, `git_status`, and
`git_diff`; it cannot write files or run commands).
Reasoning agent run metadata records the requested role, selected model, and
fallback reason without storing prompt content.

## Memory

Memory is local SQLite plus a local vector index:

```bash
april memory search "project preference"
april memory delete MEMORY_ID
april memory export
april conversation delete CONVERSATION_ID
```

Durable memory is not created automatically from every message. Explicit
requests such as "remember..." or `POST /memory` use the local `remember_memory`
Level 2 flow, reject sensitive-looking content by policy, and deduplicate exact
content/type/project repeats. Project-scoped memory search/export can be
filtered by `project_id` so unrelated projects stay isolated.

When the brain supplies `memory_queries`, APRIL retrieves local memories by policy and includes them in the agent prompt under a clearly marked context section. General planning requests also receive a small set of recent durable memories. Coding requests with a selected indexed project retrieve project-scoped vector chunks with local citations.

### Embeddings

The vector index defaults to a deterministic, dependency-free **hashed-token**
embedding. To use real **runtime-local** semantic embeddings served by a local
GGUF model through April Runtime:

```bash
# 1. Register a local embedding-role GGUF model
run april model import --role embedding --id april-embedding \
  --path models/your-embedding-model-q4_k_m.gguf

# 2. Select the runtime-local provider
export APRIL_MEMORY_EMBEDDING_PROVIDER=runtime-local
export APRIL_MEMORY_EMBEDDING_MODEL_ID=april-embedding   # optional; auto-detected

# 3. Rebuild the index under the new provider
run april memory reindex
```

The embedding model is loaded as its own dedicated instance (a chat model
cannot also embed) and is exempt from chat-specialist load/eviction limits, so
enabling it does not evict your coding or reading models.

Graceful degradation is built in: if `embedding_provider=runtime-local` is set
but no embedding-role model is registered (or the runtime reports it
unavailable), APRIL logs and audits a clear note and **falls back to
hashed-token embeddings** instead of crashing.

Switching embedding providers changes the vector space, so APRIL refuses to
silently mix spaces: searches/writes against an index built with a different
provider/dimension raise an actionable error pointing you to
`run april memory reindex`. Reindexing re-embeds existing memories and known
sources under the current provider — it never wipes your index without this
explicit command.

Document ingestion is offline. Text/source files are supported by default; PDF
text extraction is local and optional via `pip install -e '.[documents]'`.
Unsupported binary formats are reported as unsupported instead of being decoded
as arbitrary text. OCR, cloud parsing, DOCX, and HTML extraction are future
extensions.

## Voice

Voice is optional and disabled by default. Configure local `whisper.cpp`,
Piper, optional `sounddevice`, and optional openWakeWord model paths in
`configs/april.yaml` or environment variables. No voice model, speech model,
wake-word model, or binary is downloaded by APRIL.

```bash
april voice health
april voice devices
april voice ptt
april voice listen
```

Push-to-talk starts only from explicit CLI invocation. Wake-word mode is also an
explicit command and falls back to push-to-talk behavior when wake-word support
is unavailable. API startup never activates the microphone.

## Proactive Scheduler

The scheduler is optional and **off by default**. When enabled it runs a
background poll loop that fires due reminders through a notification sink, and an
optional daily briefing summarizing open tasks, reminders due in the next 24
hours, and the project count. It is never activated implicitly by API startup;
both the loop and briefings stay inert unless explicitly enabled in settings.

Enable it in `configs/april.yaml` (or the matching `APRIL_SCHEDULER_*`
environment variables):

```yaml
scheduler:
  enabled: true            # start the background reminder loop
  poll_interval_seconds: 30
  notification_sink: log   # "log" (logs/scheduler.log) or "macos" (native banner)
  briefing_enabled: true   # fire a daily briefing
  briefing_time: "08:00"   # local time, once per local day
  repo_monitor_enabled: true  # add read-only repo activity to the briefing
```

When `repo_monitor_enabled` is true, the briefing appends a read-only "Project
activity" section listing registered git projects with new commits or uncommitted
changes since the last briefing (all git access is local; `run april briefing`
previews this without advancing the baseline).

The daily briefing is restart-safe: the last briefing date is persisted, so it
fires at most once per local day even if the process restarts. You can preview
today's briefing on demand at any time, regardless of whether the scheduler is
enabled:

```bash
run april briefing
```

This calls the authenticated `GET /scheduler/briefing/preview` endpoint and
renders the title and body. `GET /health` reports the scheduler block
(`enabled`, `running`, `briefing_enabled`, `fired_reminders`).

## Quality Gates

```bash
make test
make lint
make typecheck
make check
run april config validate
run april verify --fake
run april verify --target-mac
```

Tests use fake model/audio components and do not require GGUF files, network access, microphones, speakers, whisper.cpp, Piper, openWakeWord, or `llama-cpp-python`.

`run april verify --fake` is a release smoke gate. It checks project-bound
conversations, immutable patch application, tampered artifact rejection, repo
override rejection, forced command working directories, audit records,
tool-call rows, and exactly one runtime streaming usage event.

`run april verify --target-mac` is the local target-laptop checklist. It reports
pass, fail, skip, and manual-check rows for architecture, Python, llama.cpp
availability, configured GGUF readability, real model load/chat/stream/unload
when a local model is supplied, voice dependencies, and push-to-talk smoke
steps. It never downloads models or changes system settings. Use
`--require-real-model` with a GGUF path or `APRIL_TEST_GGUF_PATH` when missing
real-model support should fail the command.

## Security Model

- Model output is advisory only.
- Unknown tools are denied.
- Permission level and risk are computed deterministically from tool policy and arguments.
- Every tool call runs through a trusted `ToolExecutionContext` containing the
  request, actor, agent, selected project, approval, and audit correlation. For
  project tools, APRIL derives the repository root from the registered project;
  model-supplied roots cannot override it.
- Level 3 and above operations require exact-action one-time approvals.
- Filesystem access is restricted to configured roots and rejects traversal, symlink escapes, sensitive locations, binary files, and oversize reads.
- Sensitive file names such as `.env`, `.env.*`, `.netrc`, private keys,
  credential files, browser credential stores, keychains, and `data/april.db`
  are denied case-insensitively.
- Subprocess execution uses argv arrays with `shell=False`; pipes, redirects,
  substitutions, shell interpreters, package installers, arbitrary `python -m`
  modules, and shell metacharacters are denied.
- External actions are disabled by default and not simulated.
- `open_app` is Level 4 and can only open configured macOS application names
  with `/usr/bin/open -a` after exact approval.
- `open_url` is Level 5, requires `external_actions_enabled`, accepts only
  normalized `http`/`https` URLs without credentials, and requires exact
  approval.

## Limitations

- The MVP fake backend is deterministic and not intelligent.
- The default vector embedding is a lightweight hashed-token baseline. Real semantic embeddings are available by registering a local embedding model and setting `embedding_provider=runtime-local` (see Memory → Embeddings).
- Desktop UI is documented as a future surface.
- The global launcher starts Runtime and the Core API. Desktop UI remains a
  future surface.
- Real wake-word, STT, and TTS require user-installed local binaries/models.
- Real GGUF inference requires manually installed model files and the optional `llama-cpp-python` dependency.
