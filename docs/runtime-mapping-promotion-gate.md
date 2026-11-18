# Runtime Mapping Promotion Gate

The sandbox discovery pipeline records evidence; it does not authorize runtime mutation. `mapping_promotion_gate.py` is the separate promotion boundary.

## Required evidence

A capability record is promoted only when all of the following are true:

1. The discovery report is schema version 1, complete (`status=measured`), and bound to the exact expected model identity.
2. Every candidate promotion is marked verified and has a deterministic experiment ID derived from identity, category, and path.
3. Every promotion has a measurement signature, and an independent replay result exists with `status=passed` and the identical measurement signature.
4. The regression report is passed, has no listed regressions, contains finite baseline/post metrics, matching metric names, and explicitly non-regressed metric rows.
5. The capability envelope’s schema signature and capability digest validate exactly.

The schema signature is a deterministic SHA-256 contract identifier. It is integrity binding, not asymmetric signing; a deployment trust root can replace it later without changing the report shape.

## Runtime boundary

The mutation firewall now requires a promoted mapping capability in the mutation context in addition to explicit mutation authorization, an armed gate, and an active transaction. A rejected, stale, tampered, or missing capability fails closed before the mutating stage can execute.

The capability record is model-identity bound and contains only promotion IDs, replay status, regression digest, schema signature, and a capability digest. It does not grant permission to mutate a different model revision and it does not replace artifact validation or rollback.
