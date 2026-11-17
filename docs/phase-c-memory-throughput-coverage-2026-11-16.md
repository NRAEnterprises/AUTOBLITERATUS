# Phase C Memory and Throughput Profiling — Coverage and Hardware Report

## Test and coverage result

- Full suite: **173 passed, 17 skipped**.
- Project-wide line coverage: **34%** across 4,816 statements (3,185 missed).
- Coverage artifacts: `coverage-phase-c/coverage.xml` and `htmlcov-phase-c/`.
- All 20 current active callable stages were verified through the callable registry.

The project-wide percentage includes 80 unavailable prototype modules and broad optional integrations. It should not be confused with active-stage contract coverage.

## Active callable-stage coverage

| Stage | Line coverage | Covered/valid lines |
|---|---:|---:|
| `tuning.anti_router_collapse` | 67.1% | 106/158 |
| `tuning.architecture_detection` | 55.7% | 34/61 |
| `tuning.audit_trail_auto` | 88.2% | 67/76 |
| `tuning.auto_ablated_capability` | 78.0% | 32/41 |
| `tuning.auto_security_scanning` | 90.2% | 37/41 |
| `tuning.baseline_auto_pre_post` | 79.7% | 47/59 |
| `tuning.batch_size_throughput_auto` | 79.7% | 59/74 |
| `tuning.benchmark_regression_auto` | 85.4% | 70/82 |
| `tuning.calibrated_quantization` | 7.9% | 9/114 |
| `tuning.latency_auto` | 80.3% | 57/71 |
| `tuning.memory_footprint_auto` | 12.8% | 5/39 |
| `tuning.model_card_generation_auto` | 31.0% | 9/29 |
| `tuning.model_export` | 21.4% | 46/215 |
| `tuning.multi_objective_auto` | 87.5% | 84/96 |
| `tuning.provenance_recording_auto` | 27.5% | 19/69 |
| `tuning.refusal_direction_ablation` | 68.0% | 17/25 |
| `tuning.reproducibility_auto` | 42.9% | 9/21 |
| `tuning.tokenization_consistency_auto` | 73.1% | 38/52 |
| `tuning.version_compatibility_auto` | 69.6% | 32/46 |
| `tuning.weight_fingerprint_auto` | 38.9% | 14/36 |

## Phase C promotion

`tuning.batch_size_throughput_auto` now exposes a report-only module-level `run(adapter, context)` contract. It profiles bounded batch sizes, reports generated tokens/second, selects the best measured candidate for diagnostic visibility, and explicitly declares `optimization_performed=False` and `model_modified=False`. It is designed to precede any mutating batch scheduler or deployment optimizer.

The existing `tuning.memory_footprint_auto` stage remains report-only and records host RSS plus accelerator allocation when available. The two reports can be compared in a later resource-budget gate.

## Remaining hardening

- Add OOM recovery and cancellation around each throughput candidate before enabling larger sweeps.
- Add KV-cache and sequence-length dimensions with quality regression gates.
- Run CUDA/x86-64/mobile and distributed profiles on target hardware; this ARM64 CPU runner cannot qualify them.
- Raise active-stage coverage for model export, quantization, memory, provenance, and hardware branches with deterministic fixtures.
- Keep optimization disabled until before/after quality, resource, reproducibility, and rollback evidence exists.
