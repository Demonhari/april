# APRIL Desktop

The desktop UI is intentionally out of scope for this MVP. The core API is designed so a future desktop app can connect over authenticated loopback HTTP without direct access to model bindings or tools.

Stable Core API contracts for a future desktop app:

- Use authenticated loopback HTTP against the Core API. Do not import model
  bindings, runtime internals, tool executors, or SQLite repositories directly.
- Read unauthenticated status from `GET /health`; it is intentionally redacted.
  Use authenticated `GET /diagnostics` only for local diagnostic screens.
- Send normal turns to `POST /chat` and streaming turns to `POST /chat/stream`.
  Preserve `conversation_id` from responses and pass `project_id` for
  project-scoped work.
- Use `GET /projects` and `POST /projects` for project selection. Repository
  paths must still resolve inside configured allowed roots.
- Use `GET /runtime/models` for model state. Do not call April Runtime directly
  from the desktop UI.
- Use `GET /approvals`, `POST /tools/approve`, and `POST /tools/deny` for the
  dedicated approval flow. A chat message such as "yes" is not approval.
- Use `POST /memory`, `GET /memory/search`, `GET /memory/export`, and
  `DELETE /memory/{memory_id}` for explicit durable local memory management.
  The UI should show exactly what is stored and should never silently create
  memories from ordinary chat.
- Use `GET /reminders`, `POST /reminders`, `DELETE /reminders/{reminder_id}`,
  and `GET /tasks` for local reminders and inspectable task plans.
- Use `POST /voice/input` for voice turns after local capture/transcription.
  Voice setup and permissions remain local user-controlled operations.
