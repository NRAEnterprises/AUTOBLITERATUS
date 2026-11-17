# Polyglot Backend Contract

A backend is eligible for AUTOBLITERATUS integration only when it is:

- disabled by default;
- invoked through a versioned sidecar or extension boundary;
- isolated from runtime model-weight writes;
- bounded by timeout, memory, output-cardinality, and cancellation controls;
- bound to model, toolchain, device, driver, input, and schema fingerprints;
- covered by reference-equivalence, malformed-input, reproducibility, and rollback tests;
- able to emit structured failures without raw prompts or secrets; and
- enabled only for an explicitly named pipeline.

The presence of a compiler, interpreter, or runtime on `PATH` is not evidence that a backend is qualified.
