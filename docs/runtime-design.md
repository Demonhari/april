# Runtime Design

April Runtime exposes:

- `POST /runtime/chat`
- `POST /runtime/stream`
- `POST /runtime/models/load`
- `POST /runtime/models/unload`
- `GET /runtime/models`
- `GET /runtime/health`

Models are loaded only by registered ID from `configs/models.yaml`. Paths are not accepted through public APIs.
Model entries may configure CPU-safe llama.cpp options such as thread count,
batch sizes, mmap/mlock, GPU layer count, explicit chat format override,
idle-unload timeout, and priority. Defaults do not require Metal or GPU
offload.

States:

- unavailable
- unloaded
- loading
- loaded
- unloading
- error

The fake backend supports deterministic normal, streaming, Brain-routing, and
structured specialist-loop responses. It is the default development and test
path and requires no GGUF files.

The llama.cpp backend is optional. It fails with a clear optional dependency error if `llama-cpp-python` is missing or the configured GGUF file is absent. Only `services/april_runtime/llama_cpp_backend.py` imports `llama_cpp`.

The Core API's `/chat/stream` endpoint forwards typed runtime stream events without waiting for a full model response.

Optional real-GGUF integration tests are controlled by `APRIL_TEST_GGUF_PATH`.
When the variable is absent, tests skip without downloading models. When set,
the test should use a small token limit and cover load, generation, streaming,
and unload against the local file only.

`run april verify --fake` exercises runtime health, model listing, structured
specialist execution, approval suspension/resume, and SSE streaming against the
fake backend. It asserts exactly one runtime `usage` event for the stream it
opens.

`run april verify --all-configured-models` (`--mac-readiness`) exercises every
configured local GGUF model that is present and readable. The Brain does not
count as passing unless load, chat, stream, unload, structured Brain JSON,
routing eval execution, and `--min-routing-accuracy` (default `0.90`) all pass.
If a strict JSON request falls back from chat/structured response formatting to
prompt completion, Runtime reports `diagnostics.structured_output_fallback: true`
and real verification blocks that Brain route.
Specialists do not count as passing unless load, chat, stream, role smoke, and
unload pass. Coding and system-action role smokes validate small JSON schemas;
all report output remains redacted to booleans and smoke kind only. Missing
optional specialists are skipped/degraded, never passed. `--require-real-model`
fails if no real configured GGUF is exercised. The compatibility field
`real_model_verified` means at least one real model passed; stronger readiness
is expressed by `verification_level`: `none`, `partial`, `core`, or `all`. Fake
or simulated runtime reports can never produce `core` or `all`.

`run april verify --soak --fake --minutes 10` is a bounded fake-backend soak
harness. It repeatedly checks health, chat, and model listing, optionally cycles
fake load/unload, records failures/latency/RSS when available, and never requires
real models or voice.

Generation options are backend-neutral for `temperature`, `top_p`,
`max_output_tokens`, stop sequences, and optional seed. Lifecycle state tracks
loaded/unloaded/error state, active requests, generation errors, recent
latency, recent tokens per second, load/unload timestamps, and the active
eviction policy. A loaded model with active requests cannot be unloaded or
evicted, and keep-loaded brain models are not idle-unloaded.

Routing first applies conservative deterministic rules for exact local
inspections, reminder operations, explicit approval IDs, and safety denials.
Ambiguous requests use the logical router model role. By default
`brain.router_model_id` is null and routing aliases `brain.model_id`, so the
registry has one model entry and Runtime loads one GGUF. A configured dedicated
router must reference a registered `role: router` model. Route provenance and
outcome-calibrated confidence are stored separately from model-generated
`BrainDecision` JSON; prompts and tool arguments are not routing diagnostics.

Prompt rendering stays centralized in April Runtime. Model config should set an
explicit `chat_format` (`granite`, `qwen`, or `generic`) where known. If absent,
Runtime first accepts trustworthy backend/GGUF chat-template metadata when
available, then falls back only for recognized Granite or Qwen model names.
Unknown models without a configured or metadata-provided template fail with a
clear unsupported-template error instead of silently using a generic format.

Context budgeting reserves configured output tokens before generation and
estimates the rendered prompt, including role/template overhead. The governing
system prompt and latest user request are required; if they cannot fit, Runtime
fails clearly. Runtime groups system messages, ordinary user turns, and
assistant-request/tool-result/assistant-continuation sequences before selection.
It prefers a supplied machine-generated conversation summary to older raw turns,
never splits an optional group, and never keeps a tool result without its
assistant request. Oversized tool results may be truncated with an explicit
marker inside the intact group; if the minimal group cannot fit, the whole group
is removed. Runtime makes no summarization model call.

Chat/stream responses expose `context_budget` metadata with estimated input
tokens, reserved output tokens, removed message/group counts, truncated
tool-result count, orphan tool-message count, complete group count, summary
inclusion, and selected context limit. Core’s per-category character ceilings
are conservative pre-bounds only; Runtime’s exact tokenizer and final context
window remain authoritative.

`GET /runtime/health` is fast and does not load models. It reports process RSS
when available, peak RSS, loaded model count, active inference count,
load/unload timestamps, last load duration, request/error counters, token
throughput, context size, threads, batch settings, and idle-unload settings. RSS
fields are process-level, not per-model physical memory; estimated values are
flagged as estimates.

Health honestly distinguishes simulation from real-model readiness. A `simulated`
boolean is `true` only for the fake backend. In that mode a missing GGUF path is
informational: the model id still appears in `missing_models`, but `status`
stays `ok` because the fake backend never loads files. With the real `llama_cpp`
backend, a missing configured path reports `status: degraded`. Genuine
backend/model errors report `degraded` in both modes, so a simulated run can
never be presented as real-model verified.

Non-keep-loaded specialist models are eligible for idle unload after their
configured timeout. When the loaded specialist count exceeds
`runtime.max_loaded_specialist_models`, APRIL evicts inactive specialists by
priority and deterministic LRU order. llama.cpp unload calls the backend
release/close method when available and clears references even after load or
generation errors.

llama.cpp streaming uses a bounded thread-safe queue bridge and reports
structured stream errors without emitting a successful completion after
producer failure.
