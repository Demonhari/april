# APRIL adaptation metadata: Colibri

Status: integration metadata only; upstream Colibri source is not staged.

APRIL's maintained integration is the typed runtime adapter at
`services/april_runtime/colibri_backend.py`. It treats Colibri as an external
local model runtime and keeps model output below APRIL's policy, approval,
audit, Tool Worker, and verification boundaries.

The adapter contract currently requires:

- an explicitly configured loopback endpoint;
- a local model directory with an exact tokenizer available to APRIL;
- no automatic service start, model download, or model activation;
- no forwarding of APRIL privileged tool definitions;
- explicit conservative resident-memory configuration for admission;
- a bounded metadata manifest rather than a synchronous full-weight hash.

Upstream repository, revision, license, and notice are intentionally unset
until the operator stages a reviewed source tree. They must be recorded in
`third_party/source-manifest.json` before this entry is changed to `vendored`.
