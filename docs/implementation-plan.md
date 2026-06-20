# APRIL Implementation Plan

## Milestones

1. Project foundation and April Runtime
   - Create Python project metadata, settings, configs, and test tooling.
   - Implement model registry validation, backend interface, fake backend, llama.cpp adapter boundary, model lifecycle, generation locking, runtime API, and streaming contract.
   - Add runtime tests that require no real model files.

2. Local security foundation
   - Implement SQLite migrations, memory repositories, audit logging, permission engine, one-time approvals, path security, read-only filesystem tools, Git read-only tools, and command policy.
   - Add security tests for path traversal, symlink escape, approval replay, command denial, and risky tool gating.

3. Brain, agents, core API, and CLI
   - Implement strict brain JSON parsing with one repair attempt and deterministic fallback routing.
   - Implement agent registry and simple bounded agent execution through April Runtime.
   - Implement core orchestrator, authenticated API, runtime model proxying, and Typer/Rich CLI.

4. Memory retrieval and repository indexing
   - Implement durable memory policy, SQLite memory operations, deterministic local hashed-token embeddings, vector index persistence, hybrid retrieval hooks, repository indexing, patch proposal boundary, and configured test runner.

5. Optional local model and voice adapters
   - Implement isolated `llama-cpp-python` backend adapter with graceful missing-dependency errors.
   - Implement optional voice health, push-to-talk pipeline, whisper.cpp subprocess STT adapter, Piper subprocess TTS adapter, and fake adapters for tests.

6. Documentation and quality gates
   - Complete README and architecture/security documents.
   - Provide safe scripts for setup, runtime, API, CLI, model placement help, and repository indexing.
   - Run tests, Ruff, mypy, and required source scans.

## Architectural Assumptions

- The repository root is the default APRIL home unless `APRIL_HOME` points elsewhere.
- The core API and April Runtime are separate FastAPI processes communicating over loopback HTTP.
- Model files are referenced only by registered model IDs from `configs/models.yaml`.
- Tests and local development can use `APRIL_RUNTIME_BACKEND=fake`.
- Specialist agent loops are intentionally conservative in the MVP: tool execution is deterministic and bounded, while generated responses come through the runtime client.
- The default vector embedding is a deterministic hashed-token baseline, not a semantic local embedding model.

## Important Security Decisions

- The brain model cannot grant or lower permissions. Tool policy and argument-sensitive checks decide the authoritative permission level and risk.
- All Level 3+ actions create pending approvals and do not execute until a later exact-action approval is consumed.
- Filesystem tools resolve paths and nearest existing parents before access, block symlink escapes, reject null bytes, cap file sizes, and deny sensitive locations.
- Shell execution is restricted to configured argv allowlists. Model-controlled commands never enable shell execution.
- Audit records are append-only JSONL. Risky operations fail closed if approval or audit state fails.
- Voice is never activated by API startup and must be explicitly invoked.

## Known External Dependencies

- Base runtime: FastAPI, Uvicorn, Pydantic v2, pydantic-settings, PyYAML, HTTPX, aiosqlite, Typer, Rich, NumPy, and Jinja2.
- Development: pytest, pytest-asyncio, pytest-cov, Ruff, and mypy.
- Optional runtime: `llama-cpp-python`.
- Optional voice: `sounddevice`, `openwakeword`, local `whisper.cpp` binary, and local Piper binary/model.
- No Homebrew, model download, cloud model API, telemetry, or microphone/speaker access is required by tests.
