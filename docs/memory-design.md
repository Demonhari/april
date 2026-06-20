# Memory Design

Structured memory uses SQLite with migrations for:

- users
- projects
- memories
- conversations
- messages
- tool calls
- approvals
- tasks
- reminders
- repo indexes
- agent runs
- schema migrations

Vector memory is local and stored under `data/vector_index/`. The MVP embedding provider uses deterministic signed hashing over normalized tokens. It is stable across Python hash seeds and requires no downloads.

The hashed embedding is a baseline retrieval aid, not a semantic model.
