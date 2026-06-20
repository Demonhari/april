# Architecture

APRIL runs as two local processes:

```mermaid
sequenceDiagram
  participant Run as run april
  participant CLI
  participant API as Core API
  participant Brain
  participant Agent
  participant Runtime as April Runtime
  participant Model as GGUF Model

  Run->>Runtime: start if missing
  Run->>API: start if missing
  Run->>CLI: delegate chat/ask/status commands
  CLI->>API: POST /chat or /chat/stream
  API->>Brain: route request
  Brain->>Agent: selected agent + planned tool calls
  API->>API: deterministic permission and approval gates
  API->>API: local memory/vector retrieval
  Agent->>Runtime: model request by registered model ID
  Runtime->>Model: optional llama-cpp-python generation
  Runtime-->>Agent: typed response or SSE token stream
  API-->>CLI: AgentResult
```

Only April Runtime imports `llama_cpp`. This keeps model bindings isolated from tools, memory, and permissions.

Core API responsibilities:

- authentication
- orchestration
- permission checks
- approval flow
- memory
- project selection and repository indexing
- tool execution
- runtime proxying and token streaming

April Runtime responsibilities:

- model registry validation
- model lifecycle
- prompt/context management
- generation locking
- SSE streaming
- optional llama.cpp integration

Repository operations require an explicit selected project. The orchestrator resolves `project_id` from SQLite or validates a supplied `repo_path` against allowed roots before any repository tool or vector retrieval runs.

The optional global launcher is intentionally small: it owns only known APRIL
subcommands, uses argv-array subprocess calls, records PIDs under `data/run/`,
and writes service logs under `logs/`. It does not start desktop UI, voice,
wake-word detection, or microphone capture.

Natural chat code modification follows a patch-first boundary. The coding model
may propose a unified diff, but APRIL validates the patch target paths, stores
the exact bytes in a content-addressed APRIL artifact, and requires a Level 3
exact-action approval before `patch_applier` can apply it once. The artifact can
live under `data/artifacts/patches/` even when the selected repository is
outside `APRIL_HOME`; only patch target paths must stay inside the repository.
Approved patch application uses `git apply --check -` and `git apply -` with the
same verified in-memory bytes.

`run april verify --fake` starts isolated temporary Runtime/Core services on
dynamic loopback ports, creates a temporary external Git project, exercises
chat, repository analysis, immutable patch approval/application, approval replay
rejection, audit checks, and runtime streaming, then stops the services.
