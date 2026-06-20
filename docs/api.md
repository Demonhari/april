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

`POST /chat/stream` emits Server-Sent Events:

- `meta`
- `token`
- `approval_required`
- `usage`
- `done`
- `error`

Request bodies larger than `api.max_request_bytes` are rejected with `REQUEST_TOO_LARGE`.
