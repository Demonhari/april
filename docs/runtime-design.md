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

The fake backend supports deterministic normal and streaming responses. The llama.cpp backend fails with a clear optional dependency error if `llama-cpp-python` is missing.
