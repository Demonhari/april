# Architecture reference archive

The directories in this archive are Git-tracked upstream source snapshots for
architectural research. APRIL does not import or package them. Their exact
origins, revisions, licenses, and APRIL adaptation records are asserted by
`third_party/source-manifest.json`.

The entries are `vendored` for source-custody purposes, have no production
dependency edge, and are excluded from APRIL's Python package surface. They
remain reference source only; APRIL production code MUST NOT import them.

The archived reference names are Mnem, AgentMW, Praxos, OpenVURP, Anybridge,
Tools-Factory, Coocon, and llm-use.
