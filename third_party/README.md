# APRIL third-party source policy

This directory contains APRIL-owned provenance and adaptation metadata plus
reviewed source snapshots for third-party runtimes and architectural references.
It is not a Python package and is not a production dependency.

The checked-in manifest is `source-manifest.json`. A source entry is ready only
when it is explicitly marked `vendored`, names a full 40-character Git
revision, preserves its upstream license and applicable notices, has an
adaptation record, and matches its recorded snapshot digest. APRIL does not
clone, download, install, or update third-party source automatically.

The reviewed Colibri source is vendored at `third_party/colibri/source/`.
APRIL's maintained integration remains the adapter in
`services/april_runtime/colibri_backend.py`; the vendored tree preserves
upstream files and licenses and is the only third-party tree eligible for the
runtime build workflow. Its explicitly allowlisted local output
`source/c/qwen36` may exist after a build; it is reported by the doctor, omitted
from source digests, and remains prohibited from Git tracking.

The eight projects under `reference_sources` are complete Git-tracked source
snapshots for provenance and architectural research. They are never imported,
packaged, or used as APRIL runtime dependencies.

## Updating a reviewed snapshot

Future updates must begin with a clean, separately reviewed upstream checkout.
Verify its origin, revision, license, notices, and security/API changes; then
replace the tracked snapshot, preserve or update the APRIL adaptation record,
refresh the manifest digests, run the doctor and (for Colibri) build checks,
and review the resulting diff. Model weights, checkpoints, build directories,
and other generated artifacts remain outside Git.

Use the local commands to validate the state and build only already-vendored
source:

```text
run april third-party doctor
run april third-party build-colibri --engine qwen36 --arch native
```

Neither command accesses the network or starts a Colibri server. The build
command only runs Colibri's declared local Makefile target after the manifest
doctor passes.

The vendored build can also be invoked directly:

```text
make -C third_party/colibri/source/c qwen36 ARCH=native
```
