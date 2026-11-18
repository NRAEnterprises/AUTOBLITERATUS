# AUTOBLITERATUS Priority 3 Handoff

Generated: 2026-11-17

## Clone the correct repository

The canonical repository is:

```bash
git clone https://github.com/NRAEnterprises/AUTOBLITERATUS.git
cd AUTOBLITERATUS
git checkout main
```

The Priority 3 checkpoint is commit `8d7cfc721322003955c7d619406e861b1d7e3305`. Verify the checkout with:

```bash
git rev-parse HEAD
git remote -v
```

Do not use the old handoff tarball or any root-level prototype copies as the active source tree. The packaged source of truth is `src/autobliteratus/`.

For the authoritative function-first status and remaining work, see [the current pipeline audit](function-first-pipeline-audit-2026-11-17.md).

## Current state

- Release posture: `conditional_not_production_qualified`.
- Runtime registry: 100 toggles; 22 callable; 78 unavailable prototypes.
- Mutating stages are opt-in and fail closed without explicit authorization.
- Mutation execution requires a transactional checkpoint and a validated promoted mapping capability.
- Sandbox discovery runs outside the runtime pipeline and stores unknown-factor evidence in a separate dynamic mapping history.
- Priority 3 promotion validates model identity, schema signature, deterministic experiment IDs, measurement signatures, replay results, finite regression metrics, and capability digest integrity.
- Visual observatory artifacts remain read-only inspection/proposal surfaces.

## Validation performed

```bash
PYTHONPATH=src python -m pytest -q
python -m ruff check src tests benchmarks
python -m build
```

The final resumed validation after the cleanup and Priority 3 changes is **261 passed and 17 skipped**. Re-run the commands above after cloning because optional quantization and hardware-dependent tests may be skipped on a different environment.

## Repository organization

- `src/autobliteratus/`: canonical package implementation.
- `tests/`: deterministic unit and integration tests.
- `benchmarks/`: benchmark runners and retained benchmark result evidence.
- `docs/`: current audits, safety contracts, historical evidence, and handoff reports.
- `configs/`: runtime configuration manifests.
- `ui/model-mapping-observatory/`: read-only visual inspection surface.
- `dist/`, caches, virtual environments, model checkpoints, and runtime output directories are generated and not tracked.

Legacy root-level duplicate Python scripts, tracked model checkpoints, generated ablation workspaces, fingerprints, provenance scratch output, reproducibility scratch output, and cross-run placeholders were removed in this cleanup checkpoint because they were not packaged, imported, or required by the current tests.

## Next engineering priorities

1. Expand the promotion gate into a complete quality/resource coordinator covering capability, safety, latency, throughput, memory, KV-cache, and resource-budget reports.
2. Complete real delta-weight artifacts, reload verification, manifest signing, and post-publish verification.
3. Add asymmetric trust-root signing for capability and artifact manifests; the current schema signature and digest are integrity checks, not cryptographic signatures.
4. Keep unavailable prototypes disabled until each receives a callable contract, bounded execution, deterministic fixtures, negative tests, and explicit mutation status.
5. Qualify CUDA, x86-64, edge, distributed, and Quanto paths on their target environments; CPU ARM64 evidence cannot establish those claims.

## Safety rule

Never treat a discovery record, topology selection, or visual graph node as mutation authority. Only a validated, model-identity-bound, replayable, regression-passing promoted capability may reach the mutation firewall, and the firewall still requires explicit operator authorization and an active transaction.
