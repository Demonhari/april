# API

Core API:

- `POST /chat`
- `POST /chat/stream`
- `POST /voice/input`
- `POST /agents/run`
- `POST /tools/request`
- `POST /tools/approve`
- `POST /tools/deny`
- `GET /approvals`
- `POST /memory`
- `GET /memory/search`
- `GET /memory/export`
- `DELETE /memory/{memory_id}`
- `DELETE /conversations/{conversation_id}`
- `GET /reminders`
- `POST /reminders`
- `DELETE /reminders/{reminder_id}`
- `GET /tasks`
- `GET /projects`
- `POST /projects`
- `POST /projects/{project_id}/index`
- `GET /runtime/models`
- `GET /health`
- `GET /diagnostics`

`/health` is unauthenticated and redacts local filesystem paths. Mutation,
memory, and diagnostic endpoints require:

```http
Authorization: Bearer APRIL_API_TOKEN
```

Stable error shape:

```json
{
  "error": {
    "code": "APPROVAL_REQUIRED",
    "message": "This action requires approval.",
    "request_id": "..."
  }
}
```

`POST /chat` and `POST /chat/stream` accept:

```json
{
  "message": "April, inspect this repository",
  "conversation_id": "...",
  "project_id": "...",
  "repo_path": "/absolute/path/inside/allowed/roots"
}
```

Repository tasks should pass `project_id` or `repo_path`. If neither is supplied, APRIL returns a clean project-selection message instead of guessing a repository.

If `conversation_id` is omitted, APRIL creates a local conversation and returns
the ID in `result.conversation_id`. Clients can pass that ID on later turns to
reuse bounded local conversation history.

Conversation IDs are project-bound. Reusing a conversation with a different
`project_id` returns a stable permission error instead of silently switching
scope.

`POST /chat/stream` emits Server-Sent Events:

- `meta`
- `routing`
- `agent_iteration`
- `tool_request`
- `tool_result`
- `token`
- `final_answer`
- `approval_required`
- `usage`
- `done`
- `error`

Request bodies larger than `api.max_request_bytes` are rejected with `REQUEST_TOO_LARGE`.

`POST /memory` stores explicit durable local memory only:

```json
{
  "content": "I prefer concise answers",
  "memory_type": "preference",
  "project_id": "optional-project-id",
  "source_conversation_id": "optional-conversation-id",
  "reason": "explicit user request"
}
```

`memory_type` is one of `fact`, `preference`, `project`, or `note`. Sensitive-looking
values such as tokens, passwords, API keys, credentials, and private keys are rejected.
Exact duplicate content/type/project writes return the existing memory record. `GET
/memory/search` and `GET /memory/export` accept optional `project_id` filters so
project-scoped memories stay isolated from unrelated projects.

Specialist `/chat` requests are Brain-routed but run through
`StructuredAgentLoop` after agent selection. General Agent chat remains a direct
model response.

`POST /agents/run` is a typed direct-agent endpoint:

```json
{
  "agent": "coding_agent",
  "message": "Inspect this repository",
  "conversation_id": "...",
  "project_id": "...",
  "options": {
    "structured": true
  }
}
```

The endpoint rejects unknown agents, validates project scope for
project-required agents, and returns `ok`, `pending_approval`, `unavailable`,
or `error` in `result.status`. Reasoning requests are always available: APRIL
uses an available `reasoning` model when configured, otherwise it runs the
normal brain model in reasoning mode and records the final model choice in run
metadata.

`POST /tools/approve` preserves direct tool approval behavior. When the approval
belongs to a suspended specialist run, the response is:

```json
{
  "status": "resumed",
  "result": {
    "status": "ok",
    "final_message": "..."
  }
}
```

If a resumed run asks for another Level 3+ tool, `result.status` is
`pending_approval` and a new approval ID is returned.

`POST /voice/input` accepts the same body shape as `/chat`, including
`conversation_id`, so a push-to-talk voice loop can keep a stable conversation.

Reminder operations are authenticated:

```json
POST /reminders
{
  "content": "stand up",
  "due_at": "2026-06-21T09:00:00Z"
}
```

`GET /tasks` returns the inspectable local task rows currently stored in
SQLite. It does not execute tasks.

## Global Launcher

The `run april` launcher is a local CLI supervisor, not a public API endpoint.
It locates `APRIL_HOME`, starts missing services with:

```bash
python -m services.april_runtime.server
python -m services.api.server
```

and then delegates to the existing `april` CLI. It stores PID files under
`data/run/` and service logs under `logs/runtime.log` and `logs/api.log`.

Supported commands include:

```bash
run april
run april --fake
run april status
run april stop
run april restart
run april logs
run april ask "April, plan my work today."
run april health
run april models
run april approvals
run april approve APPROVAL_ID
run april deny APPROVAL_ID
run april config validate
run april config inspect
run april agent run coding_agent "Inspect this repository" --project-id PROJECT_ID
run april reminder list
run april reminder create "stand up" --due-at 2026-06-21T09:00:00Z
run april reminder delete REMINDER_ID
run april task list
run april voice health
run april voice devices
run april voice ptt
run april voice listen
run april verify --fake
run april verify --target-mac
run april setup tokens --output .env
```

`--fake` affects only newly started child services by setting
`APRIL_RUNTIME_BACKEND=fake`; it does not edit configuration files.

`run april config validate` validates YAML shape, model references, agent
references, tool references, and loopback defaults. `run april config inspect`
prints effective non-secret config with the API token redacted. `run april
setup tokens --output .env` generates random local API/Runtime tokens, writes
them only to the chosen local env file, and does not print full token values.
`run april verify --fake` uses isolated temporary data paths and dynamic ports so it can
exercise the local structured specialist workflow without modifying user
projects or requiring GGUF files.
`run april verify --target-mac` reports pass, fail, skip, and manual-check rows
for target laptop validation. It never downloads models or changes system
settings. Add `--require-real-model` with a local GGUF path or
`APRIL_TEST_GGUF_PATH` when model load/chat/stream/unload checks should be
required.

`POST /tools/request`, `POST /tools/approve`, orchestrator planned tools, and
project indexing share the same trusted execution service. Repository roots and
command working directories are derived from selected projects rather than
model-provided paths.
