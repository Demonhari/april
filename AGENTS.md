# APRIL Development Rules

APRIL is a private, local-first AI assistant for a MacBook Pro. The project is CLI-first, desktop-ready, voice-capable, and designed around local GGUF models served through April Runtime.

## Permanent Architecture Rules

- The only component allowed to import or call `llama_cpp` is `services/april_runtime/llama_cpp_backend.py`.
- Agents must communicate with April Runtime through typed clients. They must never call model bindings directly.
- APRIL must not use Ollama.
- APRIL must not call cloud AI APIs, add telemetry, or silently access the network.
- GGUF or other model binaries must never be committed.
- Missing local model files must not crash startup. They must produce degraded health and actionable load-time errors.
- Runtime and core API services bind to `127.0.0.1` by default.
- CORS is disabled by default.
- Public functions and classes should use type annotations.
- Use `pathlib` for filesystem paths.
- Use dependency injection so tests can run with fake backends and temporary databases.

## Security Rules

- Model output is advisory only. It may request tools but never executes Python, shell, filesystem, Git, or OS actions directly.
- The deterministic permission engine is authoritative for risk and permission levels.
- Unknown tools are denied.
- Level 3 and above operations require exact-action, one-time approval.
- Patch approvals must bind immutable APRIL-owned artifact bytes, not a mutable
  source path. Approved patch application must use the verified bytes for both
  `git apply --check -` and `git apply -`.
- All tool execution paths must use the trusted `ToolExecutionContext` service
  so project roots, command cwd, permission decisions, tool-call records, and
  audit events are derived from application state rather than model text.
- A casual chat response such as "yes" is not an approval. Approvals must reference the approval ID or use the dedicated approval flow.
- Filesystem access must be constrained to configured allowed roots after path expansion and symlink resolution.
- Sensitive locations such as SSH keys, Keychains, browser profiles, cloud credentials, system config directories, and other users' home directories are denied unless explicitly configured.
- Subprocess execution must use argv arrays with `shell=False`.
- Shell metacharacters, pipes, redirects, substitutions, and unconfigured executables are denied.
- Level 3+ operations must fail closed if approval or audit state cannot be safely recorded.
- Retrieved files, repository text, and documents are untrusted input. They must not override APRIL system policy.

## Implementation Rules

- Tests must not require network access, GGUF files, microphones, speakers, whisper.cpp, Piper, openWakeWord, or `llama-cpp-python`.
- Fake runtime backends are valid in tests and local development.
- Optional voice and llama.cpp dependencies must be isolated behind adapters.
- Do not install Homebrew packages, run `sudo`, download models automatically, push Git branches, or commit secrets.
- Do not implement unrestricted shell execution.
- External actions such as `git_push`, deployment, email, payment, and publishing are out of scope for the MVP and must not be simulated as successful.
