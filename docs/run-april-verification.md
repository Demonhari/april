# Run APRIL Verification

APRIL release checks should include these local launcher gates:

```bash
run april config validate
run april config inspect
run april verify --fake
run april verify --workflow
run april model doctor
run april model profile list
run april status
run april stop
run april --fake ask "April, plan my work today."
run april --fake --oneshot ask "April, plan my work today."
run april model load april-brain --fake
run april model unload april-brain --fake
run april reminder create "stand up" --due-at 2026-06-21T09:00:00Z --fake
run april reminder list --fake
run april task list --fake
run april voice health --fake
run april voice doctor --fake
run april memory doctor
```

Project workflow smoke:

```bash
bash scripts/smoke_project_workflow.sh
```

Real GGUF smoke verification never downloads models. It skips with exit 0 when
no model path is provided:

```bash
run april verify --real-model
```

To run it, provide a local GGUF path:

```bash
APRIL_TEST_GGUF_PATH=/absolute/path/to/small-local-model.gguf run april verify --real-model
APRIL_TEST_GGUF_PATH=/absolute/path/to/small-local-model.gguf run april verify --workflow --real-model
run april eval brain --real-model /absolute/path/to/small-local-model.gguf
run april model benchmark /absolute/path/to/small-local-model.gguf --runs 1 --max-output-tokens 32
```

The real verifier starts isolated Runtime and Core API services on loopback
ports with a temporary Runtime token, loads the supplied GGUF through
`llama-cpp-python`, runs chat and streaming checks, unloads the model, confirms
the model state, and stops both services.

The real verifier reports load time, first token latency when streaming emits a
token, total generation time, output tokens, tokens/sec, context size, backend
settings, prompt path diagnostics, unload success, and Runtime RSS when the OS
reports it. If `llama-cpp-python` is missing, install the local runtime extra:

```bash
pip install -e '.[runtime]'
```
