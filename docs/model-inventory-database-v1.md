# Model Inventory Database v1

## What this increment provides

`autobliteratus.core.model_inventory` adds a local SQLite inventory and evidence pipeline that is independent of the curated model-family mapping table. It inventories registered modules and state-dict tensors, creates snapshot-scoped stable IDs, stores optional execution and FX graphs, emits low-authority role candidates, runs an adapter-driven isolated probe battery, and scores role predictions on held-out architecture groups.

The inventory database is append-only through SQLite update/delete triggers. Each record carries a schema version. Initialization compares the stored schema digest with the checked-in v1 schema and rejects incompatible layouts instead of silently reusing them. This prevents ordinary application updates; it is not cryptographic tamper-proofing. A process with filesystem access can still alter or replace the database. Signing, trust roots, and migration tooling are not implemented.

Tensor values are not written to the database. Their content hashes, shape, dtype, layout, strides, state-dict name, and observed module binding are recorded. If a value cannot be hashed using the available representation, the snapshot is marked `partial`; the scanner does not substitute a guessed value hash.

## Stable IDs and what they mean

`structure_id` is derived from canonical config, registered module paths/types/signatures, and tensor names/shapes/dtypes. `snapshot_id` additionally binds the source revision, metadata, model class, and per-tensor content hashes. `module_id` and `tensor_id` are derived within that snapshot, so repeated scans of identical content reproduce the IDs. A changed model state produces a different snapshot and new object IDs. These identifiers do not claim that objects are the same across revisions.

Shared module paths are retained as aliases and point to a single module instance where the model API exposes that relationship. Tensor names are kept separately, and potential state aliases are recorded only when detectable from observed parameter objects. Pointer addresses are not persisted as IDs.

## Computing-graph evidence

`capture_execution_graph()` installs temporary module hooks on a caller-supplied disposable model and records observed nested calls plus input/output tensor metadata. It does not serialize input values. Its mutation guard detects changes in `state_dict`, but not arbitrary Python-side caches, files, network calls, or other external effects. Use a disposable copy even when the guard is active.

`capture_fx_graph()` optionally records PyTorch FX operations and dataflow edges. Symbolic tracing may fail or may not represent dynamic control flow, custom kernels, or runtime-specific behavior. A successful FX trace is therefore evidence about one symbolic path, not proof that all model behavior was captured. Compare it with concrete call traces over multiple input regimes before relying on it.

`infer_structural_roles()` uses module types/names and observed graph participation to emit candidates such as attention components, normalizers, embeddings, recurrent/stateful components, and projections. The role detector is deliberately open-world and family-name independent. It currently includes one graph-pattern candidate for attention operators with dataflow from at least three distinct module calls. Other roles are heuristic candidates, not complete semantic labels. Every role assertion remains `candidate`; graph or naming clues alone never become modification permission.

## Isolated layer probes

`run_layer_probe_battery()` assesses every inventoried non-root module with five required profiles by default:

- `output_sensitivity` compares baseline and ablated output signatures and records output distance.
- `capability_axes` compares measured reasoning, general-capability, and cohesion scores.
- `input_regime_robustness` requires baseline and ablated measures for at least two regimes.
- `resource_impact` records baseline and ablated latency and peak memory.
- `stability_recovery` records baseline stability, post-removal stability, stability after a recovery procedure, recovery steps and method, a balance score and its criterion, and reasoning/general/cohesion balance-axis results.

All five profiles are required for a complete layer assessment; the API rejects a reduced profile set. Each attempted profile must provide its required structured measurements. An adapter must implement `sandbox_layer_probe(module_record, profile, context)` and return an object shaped like `{"isolated_copy": true, "removal_performed": true, "removed_module_id": "<the supplied module_id>", "measurements": {...}}`. The runner requires isolation attestation and an exact match to the snapshot-scoped module ID; a missing or mismatched removal confirmation remains unknown and contributes no measured evidence. The adapter is responsible for making a disposable copy, removing or disabling only the named module in that copy, running baseline/ablation measurements, and performing its declared recovery procedure on the ablated copy. Missing metrics, an absent isolation attestation, or missing adapter support are recorded as unknown. Probe errors are recorded as failed. User-supplied labels and permission fields are stripped from measurements.

Large models can exceed the bounded sweep. The default ceiling is 100,000 modules and 24 hours, with caller-overridable bounds. Truncation is persisted as `partial`; it must not be reported as an all-layer result. The run also records adapter type/version and probe-suite/dataset IDs, defaulting to `unversioned` when the adapter does not provide them. Role assertions link to their module's probe IDs and measurement signatures, but remain `candidate`: a measured causal effect confirms impact, not semantic identity. The battery does not automatically rebalance or edit weights. There is no universal stability or balance objective, so the adapter must identify and measure its criterion. `high_dependency_risk` warnings report observed capability changes; they do not block or authorize later removal.

## End-to-end API

```python
from autobliteratus.core.model_inventory import build_model_inventory

report = build_model_inventory(
    model,
    "artifacts/model-inventory.sqlite",
    source_revision="local-checkpoint@revision-or-digest",
    metadata={"framework": "pytorch", "framework_version": "..."},
    isolated_model=disposable_copy,
    invoke=lambda candidate: candidate(example_inputs),
    probe_adapter=adapter,  # optional; must implement sandbox_layer_probe
)
```

Pass `isolated_model` only when it is a separate disposable copy with the same config and tensor state as `model`; the orchestrator verifies the snapshot identity. `invoke` is optional. FX capture is enabled by default when a separate model copy is supplied. For a structural inventory without graph or probe evidence, omit the copy and adapter. The report always sets `model_modified=false` and `runtime_permission_granted=false`.

For evaluation, `evaluate_unseen_architectures()` requires independently annotated gold labels, predicted labels, the full held-out target universe, and group IDs that are disjoint from training architecture groups. It reports per-role precision, recall, F1, and false-positive rate, plus target coverage. If the target universe is incomplete, false-positive rates are not meaningful. Evaluation results can be stored with `persist_evaluation()`.

## Qualification still required

The current validation uses a small Python fixture with an uncatalogued toy architecture. It checks stable IDs, SQLite append-only triggers, call-trace recording, candidate generation, multi-profile probe validation, risk warnings, and held-out metric calculations. It is not evidence of real-model accuracy or successful intervention across arbitrary architectures.

This sandbox has no PyTorch or Transformers installed, so no real PyTorch FX trace, Transformers checkpoint scan, accelerator run, or post-ablation recovery experiment was performed. The FX path and adapters must be qualified in a provisioned model environment. Before using labels for model work, build a diverse, independently annotated holdout corpus across model families and frameworks; report per-role precision/recall and unsafe false-label rate; test multiple concrete inputs and modes; and require exact rollback and reload checks on disposable copies. This inventory and its role assertions remain separate from the existing runtime promotion gate and cannot authorize mutation.
