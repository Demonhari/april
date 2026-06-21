# Run APRIL Verification

APRIL release checks should include these local launcher gates:

```bash
run april config validate
run april config inspect
run april verify --fake
run april status
run april stop
run april --fake ask "April, plan my work today."
run april model load april-brain --fake
run april model unload april-brain --fake
run april reminder create "stand up" --due-at 2026-06-21T09:00:00Z --fake
run april reminder list --fake
run april task list --fake
run april voice health --fake
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
```

The real verifier starts isolated Runtime and Core API services on loopback
ports with a temporary Runtime token, loads the supplied GGUF through
`llama-cpp-python`, runs chat and streaming checks, unloads the model, confirms
the model state, and stops both services.
