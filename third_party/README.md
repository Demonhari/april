# APRIL third-party source policy

This directory contains APRIL-owned provenance and adaptation metadata for
third-party runtimes and architectural references. It is not a Python package
and is not a production dependency.

The checked-in manifest is `source-manifest.json`. A source entry is buildable
only when it is explicitly marked `vendored`, names a full 40-character Git
revision, and includes the upstream license and notice files. APRIL does not
clone, download, install, or update third-party source automatically.

The current checkout intentionally contains no Colibri upstream source. The
maintained APRIL integration is the adapter in
`services/april_runtime/colibri_backend.py`; `third_party/colibri/` records the
boundary and the metadata required before upstream source can be staged. The
doctor therefore reports Colibri as not staged until an operator supplies the
source and provenance.

The architecture projects listed in `reference_sources` are provenance-only
references. APRIL does not copy or import their source, and they cannot become
runtime dependencies. This is required by APRIL's development rules.

## Source staging contract

An operator who has separately obtained and reviewed an upstream source may
stage it under the manifest's relative path, update the entry to `vendored`,
record the exact upstream revision, and add the upstream license and notice
files. The APRIL adaptation record must be updated at the same time. Model
weights, checkpoints, build directories, and other generated artifacts remain
outside Git.

Use the local commands to validate the state and build only already-staged
source:

```text
run april third-party doctor
run april third-party build-colibri
```

Neither command accesses the network or starts a Colibri server. The build
command only runs the declared local CMake entrypoint after the manifest doctor
passes.
