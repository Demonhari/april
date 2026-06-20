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
