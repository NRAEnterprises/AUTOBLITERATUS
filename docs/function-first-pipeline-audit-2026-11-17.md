# AUTOBLITERATUS Function-First Pipeline Audit

**Audit date:** 2026-11-17  
**Repository:** `NRAEnterprises/AUTOBLITERATUS`  
**Synchronized head:** `e10b43d`  
**Scope:** topology safety export, sandboxed model discovery/testing, runtime modification/export pipelines, and remaining build priorities.

## Executive assessment

The repository is now synchronized with the current Phase C handoff and the observatory UI. The synchronized Python validation completed with **242 passed and 17 skipped tests**; the observatory production build completed successfully. The package currently discovers **100 runtime toggles**, of which **22 expose callable module-level stages** and **78 remain unavailable prototypes**. The model loader advertises **26 model types**, and the mapping registry contains all 26 with all 10 mapping categories explicitly represented. This is strong contract scaffolding, but it is not equivalent to full model-family qualification or production-safe mutation.

The most important functional risk is the default runtime posture: `tuning.refusal_direction_ablation`, `tuning.architecture_detection`, and `tuning.model_export` are enabled by default, while baseline/regression, rollback, marker creation, and audit stages are not default-enabled. A normal non-dry run can therefore mutate an in-memory model and export it without a mandatory pre/post quality gate or integrated rollback checkpoint. The next work must close that mutation firewall before expanding discovery breadth or visual integration.

## 1. Topology export and sandbox-safety audit

### Verified protections

`ui/model-mapping-observatory/exports/current-3d-mind-map-state.schema.json` requires a fixed export kind, model identity, viewport state, nodes, edges, mind-map branches, and backend context. Its safety definition uses hard schema constants:

- `direct_weight_mutation` must be `false`;
- `downstream_mode` must be `sandboxed_proposal`;
- `requires_pipeline_validation` must be `true`.

The exported snapshot contains 12 nodes, 13 edges, four mind-map branches, and the same safety values. The UI labels the surface read-only and the topology export is metadata-only. Nodes and edges describe locations and relationships; they do not carry executable edits, weight deltas, code, or runtime stage instructions.

### Current downstream interaction

The safety flags are currently **declarative only**. Repository search found the flags in the observatory schema, snapshot, and UI export code, but no runtime importer or mutation stage consumes the topology artifact. Therefore:

1. A downstream mutation pipeline does not currently receive or validate this schema.
2. `selected_node_id`, evidence labels, and graph edges do not authorize a mutation.
3. The `direct_weight_mutation: false` flag is not a runtime capability boundary until a consumer rejects invalid or mismatched artifacts.
4. The topology model identity is a fixture identity and is not cryptographically bound to a loaded model revision, state fingerprint, mapping database entry, or current runtime adapter.
5. The browser export does not itself perform JSON Schema validation before download; it constructs the expected payload in TypeScript.

This is safe in the narrow sense that no topology-to-mutation bridge exists, but it is incomplete for a functional downstream sandbox pipeline. The correct next step is a Python-side, fail-closed topology importer that validates the schema, binds the artifact to model/config/state fingerprints, rejects unknown node IDs and unknown edge endpoints, and emits a signed or hash-bound proposal artifact. It must never call a mutating stage directly.

## 2. Model discovery and mapping pipeline audit

### What is implemented

The discovery boundary is correctly separated from the runtime pipeline:

- `src/autobliteratus/discovery_pipeline.py` is physically outside `autobliteratus.pipeline` and the runtime toggle registry.
- It requires `sandbox=True`, rejects mutation-enabled or live/runtime contexts, requires an existing `sandbox_root`, confines the mapping database below that root, and requires the adapter to be explicitly bound to the same root.
- `model_mapping_discovery.py` runs bounded read-only experiments for config identity, state inventory, candidate path resolution, shape checks, cache API/output behavior, and a state mutation guard.
- Discovery records model/config/state fingerprints and atomically appends evidence to a JSON database.
- Unknown categories remain unknown and are not promoted by user labels.
- The registry covers all 26 loader model types and all 10 declared mapping categories with explicit `verified`, `conditional`, or `unsupported` states.
- Dedicated fixtures cover discovery, unknown areas, atomic persistence, adapter binding, and runtime/discovery isolation.

### Remaining functional gaps

The mapping registry is complete as a **category inventory**, not as a complete parameter-level or behavior-level map. It still needs:

1. **Topology-level resolution:** every resolved area should enumerate concrete modules, state-dict keys, tensor shapes, dtype/device, tied-weight relationships, and parent/child connections rather than only broad candidate paths.
2. **Family-specific evidence:** a static mapping entry is still a hypothesis until the actual model instance passes runtime checks. The current promotion unit is an area/category, not every weight-bearing subcomponent.
3. **Unknown-model experiment controls:** the discovery battery validates API and state behavior but lacks a first-class timeout/cancellation/resource budget around every probe. State fingerprinting can be expensive for large models.
4. **Database integrity:** the mapping database is atomically written but not schema-validated, signed, chained, deduplicated, or protected against replay of stale evidence. It should bind each record to model revision, config hash, state fingerprint, adapter/toolchain/device metadata, and experiment configuration.
5. **Promotion consumer:** evidence is persisted, but no runtime gate consumes a promoted mapping record to authorize a surgical modification. This is appropriately fail-closed but means the mapping pipeline is not yet end-to-end functional.
6. **Experiment isolation depth:** the adapter boundary is enforced, but a future discovery worker should run in a separate process/container with filesystem, network, memory, timeout, and cancellation controls rather than relying only on Python-level context flags.
7. **Test matrix:** the current fixtures are deterministic and valuable, but hardware-specific mapping behavior, sharded device maps, quantized wrappers, remote-code models, tied embeddings, and mixed architecture families still require target-runner evidence.

## 3. Testing pipeline audit

### Current status

The synchronized test suite has 30 test files, with 260 passing tests and 17 skips in the current environment. Coverage includes runtime registry contracts, discovery isolation, mapping completeness, KV-cache profiling, memory/throughput failure paths, export hardening, baseline/regression gates, provenance/fingerprint behavior, quantization compatibility, C++ SIMD equivalence, transactional rollback, and replayable mapping promotion.

The 17 skips are not proof of failure; they represent environment-dependent or unavailable paths. The handoff records that project-wide coverage was previously approximately 34%, accelerator backends are unavailable in the present runner, and Optimum Quanto/hardware matrix evidence is environment-blocked. Those constraints must remain explicit in release reporting.

### Still needed

- A required test matrix for CPU, CUDA, x86-64, and any supported edge/distributed target.
- A controlled quantization runner with the locked extra installed and full-precision versus Quanto comparisons.
- Property/fuzz tests for topology import, mapping database records, malformed paths, stale fingerprints, replay, and schema evolution.
- Process-level discovery containment tests for timeout, cancellation, memory ceiling, filesystem escape, and worker crash recovery.
- Mutation transaction tests proving that a failed quality gate leaves no published artifact and that a rollback restores the exact pre-edit fingerprint.

## 4. Modification and export pipeline audit

### What is implemented

The runtime pipeline has several solid safety components:

- Local model output paths are checked for source overlap.
- Model export uses a temporary directory, validates config/tokenizer/weights, writes a manifest with runtime and SHA-256 information, and publishes through a no-replace rename.
- Baseline and regression stages validate finite metrics and can enforce metric-specific directions/tolerances when explicitly enabled.
- Memory, latency, throughput, KV-cache, and resource-budget stages are report-only and expose machine-readable statuses.
- Calibrated quantization validates representative data, requested weight types, quantized module coverage, and calibration-file fingerprints.
- Provenance, reproducibility, audit, and weight-fingerprint stages exist as callable contracts or tested supporting modules.
- The orchestrator topologically orders dependencies and does not mark a failed stage complete, so dependent stages are skipped.

### Critical gaps

1. **Quality-gate completion remains required.** Mutating stages are now opt-in and fail closed without explicit authorization, a transaction, and a promoted mapping capability. Baseline/regression/resource evidence is still not automatically synthesized for every possible modifying configuration; Priority 3 provides the mapping capability contract, not a general quality policy.
2. **Transactional rollback is implemented for the current mutation boundary.** `MutationTransaction` snapshots state, publishes checkpoints atomically, restores on pipeline failure, and verifies exact fingerprints. The older marker and auto-rollback prototype modules remain unavailable and are not represented as production stages.
3. **Mapping authorization is now evidence-bound.** The firewall requires a schema- and digest-validated promoted capability record whose model identity, replay results, measurement signatures, and regression report have passed validation.
4. **Quantization is not transaction-safe.** `_97_calibrated_quantization.py` mutates/wraps the in-memory model before all downstream export and final quality checks complete. A later failure can leave the process with a modified in-memory model even though no artifact is published.
5. **Delta export is a stub.** Both delta export modules return a path but do not compute, serialize, validate, or fingerprint an actual delta artifact. It must not be treated as a completed modification pipeline.
6. **Failure handling remains mixed by design.** The orchestrator records independent diagnostic failures, while a modifying run now has a transaction coordinator that rolls back after mutation/export failure. Additional callable mutation stages must still adopt this boundary before promotion.
7. **Artifact manifests are not signatures.** SHA-256 checksums and provenance are useful integrity evidence, but there is no signing key, signed checkpoint, rollback token, or trust-root verification for downstream acceptance.
8. **Topology ingestion is validated but remains non-authorizing.** The observatory proposal importer validates identity, graph, and safety fields, but visual selection cannot authorize mutation. A promoted mapping capability and independent runtime quality gates remain mandatory.

## 5. Function-first build priorities

### Priority 0 — Mutation firewall and release stop condition — complete

All currently callable mutating runtime stages are opt-in and fail closed unless the caller explicitly requests a named mutation pipeline. The runtime requires explicit authorization, a pre-edit transaction fingerprint, verified mapping capability, and rollback before mutation. Full cross-domain quality/resource coordination remains a later gate.

### Priority 1 — Transactional checkpoint and rollback manager — implemented for current boundary

The current pipeline snapshots model state before mutation, binds the checkpoint to identity and fingerprints, restores on pipeline failure, and publishes checkpoint state atomically. The older marker/auto-rollback prototype stages remain unavailable; rollback tokens and trust-root signing remain future work.

### Priority 2 — Quality/resource gate coordinator — partially complete

The runtime has separate machine-readable baseline, regression, latency, throughput, memory, KV-cache, and resource-budget reports. A single required coordinator covering all evidence classes before export remains the next functional target.

### Priority 3 — Replayable mapping promotion and complete modification artifacts — mapping gate complete

Implement real delta-weight computation/export, artifact reload tests, manifest schema/signing, and post-publish verification. Ensure quantization and ablation produce a complete before/after record and cannot leave a usable artifact after failed validation.

### Priority 4 — Topology-to-proposal importer

Add a Python-side schema validator and proposal importer for the observatory artifact. Bind the exported topology to a model revision/config/state fingerprint, permit only sandboxed proposal records, reject stale/unknown/unsupported nodes, and keep proposal ingestion separate from mutation execution.

### Priority 5 — Discovery hardening and promotion

Expand runtime evidence to concrete module/key/shape/connection maps, add process-level resource controls, schema/signature/replay protection for the mapping database, and create a promotion gate that produces a versioned mapping capability record consumed by the mutation firewall.

### Priority 6 — Hardware and quantization qualification

Run the pinned CPU/CUDA/x86-64/edge matrix, install and qualify the locked quantization extra, and record driver/framework/kernel/device fingerprints. Do not infer accelerator or distributed readiness from CPU-only evidence.

### Priority 7 — Callable stage completion

Promote the remaining 78 prototypes one workstream at a time. Each stage needs a real `run(adapter, context)` contract, deterministic positive/negative tests, dependency declaration, structured failure envelope, resource bounds, and explicit mutation status. Do not bulk-enable by language or file presence.

### Priority 8 — Visual layer integration

Only after the functional gates above are enforced, connect the observatory to signed, read-only reports and proposal artifacts. Keep Three.js, mind-map interchange, and GPU context cards as inspection surfaces. They should never become a mutation authority or bypass Python validation.

## Release decision

The repository is synchronized and the diagnostic/discovery scaffolding is materially ahead of the original GitHub state. It is **not ready for unrestricted mutating production runs**. The next implementation task should be the mutation firewall plus transactional rollback, followed by the topology proposal importer and discovery promotion contract. Visual work should remain last until those functional boundaries are executable and tested.

## Implementation follow-up

The first function-priority safety increment is now implemented on top of this audit:

- `refusal_direction_ablation`, `calibrated_quantization`, and `model_export` are opt-in by default.
- The CLI exposes `--allow-mutation`; direct mutating stages also enforce the same authorization contract.
- `MutationTransaction` creates an exact state snapshot, atomically publishes the checkpoint directory, restores the original model on pipeline failure, and verifies exact fingerprint restoration.
- `topology_proposal.py` validates observatory schema/safety fields, graph references, and caller-supplied model identity before returning a metadata-only proposal envelope.
- The implementation is covered by dedicated firewall, rollback, atomic-publication, and topology-tamper tests.

Post-change validation: **248 passed, 17 skipped**, Ruff clean, package build successful. The safety work reduces accidental mutation risk but does not yet replace the separate baseline/regression/resource-budget promotion gates; those remain required for the next hardening increment.

## Priority 2 implementation follow-up

The sandbox discovery pipeline now includes a bounded unknown-factor experiment battery. It performs isolated removal probes, restores the live model after every probe, records verified/unknown/failed outcomes, enforces experiment and wall-clock bounds, and atomically stores only verified promotions under a separate `dynamic_mappings` database section. User labels are excluded from promotion evidence. A sandbox-bound adapter remains mandatory, and the runtime pipeline does not consume the dynamic database automatically.

The transaction integration suite now also injects a simulated downstream quality-gate failure and verifies exact fingerprint restoration plus a durable rolled-back checkpoint manifest. Post-change validation: **255 passed, 17 skipped**, Ruff clean.
