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

Generation options are backend-neutral for `temperature`, `top_p`,
`max_output_tokens`, stop sequences, and optional seed. Lifecycle state tracks
loaded/unloaded/error state, active requests, generation errors, recent
latency, recent tokens per second, load/unload timestamps, and the active
eviction policy. A loaded model with active requests cannot be unloaded or
evicted, and keep-loaded brain models are not idle-unloaded.

Non-keep-loaded specialist models are eligible for idle unload after their
configured timeout. When the loaded specialist count exceeds
`runtime.max_loaded_specialist_models`, APRIL evicts inactive specialists by
priority and deterministic LRU order. llama.cpp unload calls the backend
release/close method when available and clears references even after load or
generation errors.

llama.cpp streaming uses a bounded thread-safe queue bridge and reports
structured stream errors without emitting a successful completion after
producer failure.
