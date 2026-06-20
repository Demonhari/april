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
- `GET /memory/search`
- `GET /memory/export`
- `DELETE /memory/{memory_id}`
- `DELETE /conversations/{conversation_id}`
- `GET /projects`
- `POST /projects`
- `POST /projects/{project_id}/index`
- `GET /runtime/models`
- `GET /health`

`/health` is unauthenticated. Mutation and memory endpoints require:

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
- `token`
- `approval_required`
- `usage`
- `done`
- `error`

Request bodies larger than `api.max_request_bytes` are rejected with `REQUEST_TOO_LARGE`.

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

The endpoint rejects unknown agents, returns `unavailable` for agents with no
configured model such as the default Reasoning Agent, validates project scope
for project-required agents, and returns `ok`, `pending_approval`,
`unavailable`, or `error` in `result.status`.

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
run april verify --fake
```

`--fake` affects only newly started child services by setting
`APRIL_RUNTIME_BACKEND=fake`; it does not edit configuration files.

`run april config validate` validates YAML shape, model references, agent
references, tool references, and loopback defaults. `run april config inspect`
prints effective non-secret config with the API token redacted. `run april
verify --fake` uses isolated temporary data paths and dynamic ports so it can
exercise the local structured specialist workflow without modifying user
projects or requiring GGUF files.

`POST /tools/request`, `POST /tools/approve`, orchestrator planned tools, and
project indexing share the same trusted execution service. Repository roots and
command working directories are derived from selected projects rather than
model-provided paths.
