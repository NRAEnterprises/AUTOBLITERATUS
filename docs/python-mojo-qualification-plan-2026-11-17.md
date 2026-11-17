# Python + Mojo Numerical Acceleration Qualification Plan

## Current status

Mojo is not installed in the current environment. The Python side is therefore the only active implementation, and no Mojo backend is enabled or claimed as verified.

The intended pairing is not a rewrite of AUTOBLITERATUS reasoning in Mojo. Python remains responsible for prompts, model interaction, policy, evidence, provenance, and promotion decisions. Mojo may implement narrowly scoped pure numerical kernels when a scalar Python or NumPy reference exists.

## Initial scope

Start with kernels that are deterministic, model-state-independent, and easy to compare against a scalar reference:

1. Float32 affine and clamp transforms.
2. Layerwise summary statistics such as mean, variance, min, max, and finite-value counts.
3. Bounded histogram or quantile accumulation for calibration telemetry.
4. Masked reductions used by calibration and resource-budget reports.

Do not begin with model mutation, attention kernels, quantization writes, generation control, or arbitrary Python callbacks. Those areas have larger numerical, memory-layout, and safety surfaces.

## Boundary

The Python stage owns the typed input buffer, shape and stride validation, device metadata, seed, resource budget, and reference result. Mojo receives immutable numeric buffers and explicit scalar parameters. It returns a typed result buffer plus status, elapsed time, allocation counters, and a backend fingerprint.

The first implementation should be a sidecar or tightly scoped extension with an explicit contract version. It must not receive model-weight write handles, filesystem credentials, network credentials, or runtime pipeline mutation callbacks.

## Required equivalence contract

For each kernel, the qualification harness must compare:

- values and shapes;
- dtype and finite-value behavior;
- NaN and infinity rejection policy;
- overflow and underflow behavior;
- masked and empty-input behavior;
- non-contiguous or strided input behavior;
- CPU fallback behavior;
- deterministic repeated outputs; and
- numerical error against a declared absolute and relative tolerance.

The Python reference must quantize at the same precision as the Mojo kernel before comparison. A faster result is not a pass if it changes calibration decisions or produces an artifact outside the declared tolerance.

## Qualification stages

### Stage 0 — toolchain and contract

Pin the Mojo compiler, Python, NumPy/PyTorch versions, operating system, CPU ISA, and build flags. Define the typed buffer schema and structured failure schema. Record compiler and source fingerprints in every result.

### Stage 1 — scalar equivalence

Run deterministic vectors covering zero, negative, positive, boundary, subnormal, overflow-sensitive, masked, empty, and non-contiguous inputs. Compare Mojo with the scalar Python reference before measuring performance.

### Stage 2 — property and differential testing

Generate seeded random vectors and compare both implementations over multiple shapes, strides, dtypes, and parameter ranges. Include adversarial sizes around vector widths and allocation boundaries. Fail closed on unexplained differences.

### Stage 3 — resource behavior

Measure latency, allocations, peak memory, throughput, cancellation, and timeout behavior. Run the same cases under a CPU fallback and ensure Mojo failure falls back to Python without changing the run's evidence status.

### Stage 4 — calibration impact

Run representative calibration corpora through both implementations. Compare not only raw statistics but downstream calibration choices, quantization scales, resource-budget decisions, and exported manifests. Any decision divergence requires investigation, not an averaged score.

### Stage 5 — sandbox discovery

Run Mojo only in the dedicated discovery sandbox with no runtime write handles. Record model, corpus, device, compiler, driver, source, and schema fingerprints. Discovery results remain evidence and are not automatically promoted.

### Stage 6 — named opt-in backend

Only after the previous gates pass may a named Python pipeline opt into the Mojo kernel. The backend remains disabled by default and must retain a Python fallback, rollback path, and regression ledger.

## Promotion blockers

Mojo must not be promoted if the environment lacks a pinned compiler, if the Python reference is underspecified, if numerical differences alter a downstream calibration decision, if memory behavior exceeds the resource budget, if cancellation is unsafe, or if a failure cannot be represented without raw prompts or secrets.

The initial acceptance target is equivalence and safe fallback, not a headline speedup. Performance is useful only after output and decision equivalence are established.
