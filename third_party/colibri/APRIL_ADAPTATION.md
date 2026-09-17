# APRIL adaptation metadata: Colibri

Status: vendored upstream source with an APRIL-maintained integration.

Imported upstream revision: `a8f2ca623ffe9de9df11d56f34d11d2d501493d3`
Imported on: 2026-09-16
Upstream repository: https://github.com/JustVugg/colibri.git
License: Apache-2.0
Classification: runtime

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

The generated Qwen36 engine at `source/c/qwen36` is an explicitly allowlisted
local build output. It is not source, is excluded from snapshot digests, and is
kept out of Git by APRIL's scoped ignore and source-hygiene rules.

Original upstream portions remain under their original copyright and license;
the upstream `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md` files are
retained in this snapshot. Subsequent APRIL-specific modifications are
maintained in this monorepo.

Material APRIL modifications to the imported source: none. The snapshot follows
the verified upstream revision, with the tracked runtime cache
`c/deepseek_v4_tiny/.coli_usage` and binary test fixture
`c/tests/fixtures/e8_case.bin` excluded as documented in the manifest. APRIL
integration code remains outside the snapshot in
`services/april_runtime/colibri_backend.py`.
