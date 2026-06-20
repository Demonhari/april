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

`POST /chat/stream` emits Server-Sent Events:

- `meta`
- `token`
- `approval_required`
- `usage`
- `done`
- `error`

Request bodies larger than `api.max_request_bytes` are rejected with `REQUEST_TOO_LARGE`.

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
```

`--fake` affects only newly started child services by setting
`APRIL_RUNTIME_BACKEND=fake`; it does not edit configuration files.
