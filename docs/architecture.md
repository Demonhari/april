# Architecture

APRIL runs as two local processes:

```mermaid
sequenceDiagram
  participant CLI
  participant API as Core API
  participant Brain
  participant Agent
  participant Runtime as April Runtime
  participant Model as GGUF Model

  CLI->>API: POST /chat
  API->>Brain: route request
  Brain->>Agent: selected agent + tools
  Agent->>API: typed tool requests
  Agent->>Runtime: model request by registered model ID
  Runtime->>Model: llama-cpp-python generation
  Runtime-->>Agent: typed response/stream
  API-->>CLI: AgentResult
```

Only April Runtime imports `llama_cpp`. This keeps model bindings isolated from tools, memory, and permissions.

Core API responsibilities:

- authentication
- orchestration
- permission checks
- approval flow
- memory
- tool execution
- runtime proxying

April Runtime responsibilities:

- model registry validation
- model lifecycle
- prompt/context management
- generation locking
- SSE streaming
- optional llama.cpp integration
