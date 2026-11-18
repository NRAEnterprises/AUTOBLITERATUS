# Repository Cleanliness Checklist

Updated: 2026-11-17

## Canonical structure

- [x] Package source lives under `src/autobliteratus/`.
- [x] Tests live under `tests/`.
- [x] Benchmark code and retained result evidence live under `benchmarks/`.
- [x] Audits, contracts, and handoffs live under `docs/`.
- [x] Runtime configuration manifests live under `configs/`.
- [x] Visual inspection source remains under `ui/model-mapping-observatory/`.
- [x] Legacy root-level duplicate Python scripts removed.

## Generated and dead artifacts

- [x] Tracked model checkpoints removed from the source repository.
- [x] Generated ablation workspace removed.
- [x] Generated fingerprints, provenance scratch output, reproducibility scratch output, and cross-run placeholders removed.
- [x] Python bytecode, caches, build products, and local environments remain ignored and are removed before final checkpoint.
- [x] Benchmark result evidence retained only where it is a dated, reviewable artifact.

## Documentation accuracy

- [x] README counts match the current registry: 100 total, 22 callable, 78 unavailable.
- [x] README no longer claims mutation is enabled by default.
- [x] Function-first audit records Priority 0–3 implementation status.
- [x] Priority 3 promotion contract is documented.
- [x] Handoff contains the canonical clone URL and main branch instructions.
- [ ] Historical dated reports are retained only when their date and historical scope are clear.

## Functional gates

- [x] Full test suite passes.
- [x] Ruff passes over canonical source, tests, and benchmarks.
- [x] Package build succeeds.
- [x] Discovery remains sandbox-only.
- [x] Unknown and failed discovery factors cannot promote.
- [x] Replayability and measurement signatures are required for mapping promotion.
- [x] Regression evidence is required for mapping promotion.
- [x] Capability schema and digest tampering fail closed.
- [x] Mutation firewall requires a promoted capability and transaction.
- [ ] Asymmetric signatures and deployment trust roots remain future work.
- [ ] Full quality/resource coordinator remains future work.
