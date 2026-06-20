# Runtime Design

April Runtime exposes:

- `POST /runtime/chat`
- `POST /runtime/stream`
- `POST /runtime/models/load`
- `POST /runtime/models/unload`
- `GET /runtime/models`
- `GET /runtime/health`

Models are loaded only by registered ID from `configs/models.yaml`. Paths are not accepted through public APIs.

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
active requests, generation errors, recent latency, and recent tokens per
second. A loaded model with active requests cannot be unloaded. llama.cpp
streaming uses a thread-safe queue bridge and reports structured stream errors
without emitting a successful completion after producer failure.
