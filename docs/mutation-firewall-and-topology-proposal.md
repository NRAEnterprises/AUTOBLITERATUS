# Mutation Firewall, Transactions, and Topology Proposals

## Safe defaults

`refusal_direction_ablation`, `calibrated_quantization`, and `model_export` are opt-in. A runtime invocation that resolves a mutation-capable stage must pass `allow_mutation=True` or `--allow-mutation` and must explicitly enable model export. Without that authorization, the pipeline stops before loading model weights.

Each mutating stage calls `require_mutation_authorized` directly. Direct stage invocation therefore cannot bypass the pipeline-level check. The stage requires explicit mutation authorization, an armed mutation gate, and an active `MutationTransaction`.

## Transaction boundary

`MutationTransaction.begin` fingerprints the ordered model state, deep-copies the exact state dictionary, writes a durable `state.pt` and checkpoint manifest into a hidden staging directory, then atomically renames the complete directory into `mutation_checkpoints/`. A checkpoint is never published in a partially written state.

On any orchestration, audit, or commit failure, the pipeline restores the original model object and state dictionary, verifies the restored fingerprint exactly matches the pre-mutation fingerprint, and reports a rollback. On success, commit records before/after fingerprints and leaves the durable checkpoint manifest for audit and recovery tooling.

The current transaction intentionally keeps the full state snapshot in memory while writing a durable copy. A future large-model implementation can replace this with an immutable shard snapshot while preserving the same fingerprint and atomic-publish contract.

## Topology proposal importer

`autobliteratus.core.topology_proposal.validate_topology_proposal` accepts a path or mapping and validates:

- schema version and export kind;
- the exact non-mutating safety contract;
- model identity fields supplied by the runtime;
- unique node IDs and valid edge endpoints;
- the selected viewport node.

It returns a metadata-only proposal envelope with a canonical SHA-256 digest, node IDs, edge count, and `mutation_authorized: false`. The importer never executes a graph instruction or changes model state. Pipeline callers may attach a validated proposal through `stage_options.topology_proposal_path`, `stage_options.topology_expected_identity`, and optionally `stage_options.topology_sandbox_root`. A mismatched or tampered proposal fails closed before transaction start.

Topology validation is evidence binding, not a quality gate. Runtime mapping evidence, baseline evaluation, regression, resource budgets, and final artifact validation remain separate required controls for a future fully authorized modification workflow.

## Operator example

A mutation-capable run must now make the authorization visible:

```bash
autobliteratus \
  --model-path ./model \
  --allow-mutation \
  --enable tuning.refusal_direction_ablation tuning.model_export
```

This enables the transaction firewall, but does not imply that a topology proposal or quality gate was supplied. Those inputs must be added explicitly when the corresponding downstream gate is required.
