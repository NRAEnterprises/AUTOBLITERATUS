# Polyglot Runtime Pairing Plan

## Decision

The proposed language pairings are useful architectural hypotheses, but they must not be embedded indiscriminately into the Python runtime. AUTOBLITERATUS will keep Python as the policy and evidence coordinator and integrate other languages only through versioned, optional sidecars or tightly tested extensions.

A language pairing is not enabled merely because its executable exists. It must have a contract, a bounded process boundary, model-identity binding, deterministic or reference-equivalence tests, resource budgets, structured failures, and rollback evidence.

## Current environment

The current development environment has Python, Node/npm, GCC, and G++ available. Rust, Cargo, Go, Elixir, Mojo, Julia, CUDA tooling, Prolog, Mercury, Racket, OCaml, Menhir, and Clang are not installed. Therefore all pairings except the generic C++/G++ direction are currently design-level or environment-blocked. No optional toolchain is being installed as part of this change.

## Pairing allocation

| Pairing | Proposed responsibility | Safe boundary | Current status |
|---|---|---|---|
| Elixir + Rust | Supervised orchestration and fault containment | Elixir supervisor plus Rust sidecar through ports or gRPC; avoid NIFs for untrusted work | Environment-blocked |
| Python + Mojo | Reasoning control with numerical acceleration | Python owns policy and evidence; Mojo owns pure bounded kernels | Environment-blocked |
| Go + TypeScript | Routing and protocol integration | Go owns bounded routing; TypeScript owns schema-facing adapters | Environment-blocked |
| Julia + CUDA | Statistical calibration and accelerator experiments | Sandbox worker producing signed, metadata-bound artifacts | Environment-blocked |
| Prolog + Mercury | Formal logic and explainable constraints | Pure, bounded query service with capped proof depth and result count | Environment-blocked |
| Rust + Tokio | Distributed coordination | Coordinator owns leases, cancellation, and health only; no weight writes | Environment-blocked |
| C++ + SIMD | Signal processing and tensor packing | Versioned native extension or isolated worker with immutable buffers | Partially available; compiler/toolchain qualification required |
| OCaml + Menhir | Grammar parsing and plan validation | Pure parser producing a schema-validated AST | Environment-blocked |
| Racket + Lisp | Controlled meta-reflection and experiment-plan generation | Discovery-only plan generator; Python approves every plan | Environment-blocked |

The complete machine-readable version is in [`configs/polyglot-runtime-pairings.yaml`](../configs/polyglot-runtime-pairings.yaml).

## Pipeline placement

### Runtime pipeline

The runtime pipeline may consume only promoted, versioned backend contracts. It must not discover new mappings, execute arbitrary generated code, or grant foreign backends model-weight write access. Foreign backends return measurements, plans, or routing decisions; Python applies the existing quality, resource, provenance, and rollback gates.

### Discovery pipeline

The separate mapping-discovery pipeline may use experimental backends only inside a sandbox. Its outputs are evidence records bound to model, toolchain, dataset, prompt, seed, device, driver, and backend fingerprints. Discovery output does not automatically promote a runtime mapping or change a live model.

### Coordination pipeline

A future Rust/Tokio or Elixir/Rust coordinator may supervise workers, but coordination must remain separate from model mutation. A worker receives an immutable input artifact and returns a versioned result artifact. Any mutation requires a separate Python-controlled checkpoint and rollback transaction.

## Required backend contract

Every sidecar or extension must expose:

```text
backend_id
contract_version
toolchain_fingerprint
input_schema_hash
output_schema_hash
model_identity_or_none
run_id
status
metrics
failures
artifacts
resource_usage
reproducibility_metadata
```

Expected failures are data, not unstructured process output. At minimum, backends must distinguish invalid input, timeout, cancellation, resource exhaustion, unsupported architecture, numerical divergence, protocol mismatch, and internal failure.

## Safety requirements

Foreign runtimes are disabled by default. They must not receive production credentials, unrestricted filesystem access, unrestricted network access, or direct write access to runtime model directories. Discovery workers must use a dedicated process or container boundary for untrusted model code; the Python-level sandbox contract is not a substitute for OS isolation.

No backend may silently change precision, device placement, tensor layout, or model state. Such changes must appear in the result manifest and be checked against a reference implementation or an explicitly approved tolerance.

## Staged adoption order

1. Define the Python-side contract and artifact schema.
2. Prototype one low-risk pure function backend, preferably parser or scalar numerical equivalence.
3. Add subprocess/container isolation and bounded resource controls.
4. Build reference-equivalence and malformed-input batteries.
5. Bind all results to model and toolchain fingerprints.
6. Run discovery-only experiments and record failures as carefully as passes.
7. Integrate one named backend into one named pipeline behind an explicit feature flag.
8. Add rollback and regression evidence before considering broader adoption.

The project should not add all nine pairings at once. Each pairing is a separate qualification program with its own failure modes and evidence ledger.
