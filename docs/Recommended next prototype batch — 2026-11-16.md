# Prototype batch implementation record — 2026-11-16

## Review scope

The initial review snapshot had 99 toggles: 12 callable and 87 unavailable. All three selected stages below are now implemented and tested; the current registry has 15 callable and 84 unavailable. The refreshed source inventory covers every currently unavailable entry, including toggle ID, module purpose, classes/methods, and line count. Prioritization favored high release value, reuse of existing pipeline/benchmark infrastructure, deterministic local tests, and low risk of silently modifying user artifacts.

## Implemented batch (in priority order)

| Priority | Toggle | Prototype | Benefit | Implementation |
|---|---|---|---|---|
| 1 | `tuning.baseline_auto_pre_post` | `_07_baseline_auto_tuning_pre_post.py` | Foundation for a paired quality gate. | Captures finite pre-edit causal-LM metrics from an explicit evaluation battery and stores them in pipeline context; no implicit model clone or calibration-data reuse. |
| 2 | `tuning.benchmark_regression_auto` | `_81_benchmark_regression_auto_tuning.py` | Detects harmful post-edit changes and blocks export. | Compares exact matching finite metrics with declared directions; defaults to 2% relative tolerance for NLL/perplexity and 10% for noisy forward throughput, supports per-metric overrides, and raises a stage failure on regression. |
| 3 | `tuning.audit_trail_auto` | `_89_audit_trail_auto_tuning.py` | Connects stage outcomes and artifacts to a durable run record. | Writes unique JSONL and Markdown audit files atomically after orchestration, records stage failures by type, and excludes raw prompts, arbitrary options, and raw exception messages. |

The baseline and regression stages use a shared evaluation helper and retain distinct responsibilities. Selecting the regression stage automatically enables baseline capture; the baseline is ordered before model edits, and the regression stage is ordered after active ablation/quantization. Non-finite/missing/mismatched measurements fail closed. Both stages remain opt-in and must be treated as a limited NLL/perplexity/forward-throughput gate, not as evidence about hallucination, reasoning, safety, or generation latency. Pipeline and unit coverage lives in `tests/test_baseline_regression_audit.py`.

## Why some other candidates should wait

- `tuning.version_compatibility_auto` (`_70_...`) currently accepts arbitrary loader callbacks. The exporter already performs a Transformers save/reload validation; a broader compatibility stage should wait until supported framework targets and their CI environments are declared, so a missing optional backend is not misreported as a model defect.
- `tuning.tokenization_consistency_auto` (`_47_...`) compares two tokenizers, but this pipeline currently does not tune tokenizers and its exporter already validates the serialized tokenizer. It is useful if tokenizer transformations are introduced, but currently duplicates a narrow existing invariant.
- `tuning.auto_rollback` and `tuning.marker_creation` (`_25_...`, `_26_...`) can delete or restore user artifacts. Defer until the quality gate and durable checkpoint format are specified. The current exporter already stages a separate output and publishes atomically, while preserving the source.
- `tuning.quantized_export_auto` (`_93_...`) overlaps the existing calibrated Quanto path. Do not add a second exporter until backend coverage and its acceptance contract are defined.

## Inventory

The complete refreshed static inventory of 84 unavailable stages is in [`remaining-prototype-inventory-2026-11-16.csv`](remaining-prototype-inventory-2026-11-16.csv). Prototype descriptions are intent, not evidence that the existing class methods work or satisfy production criteria; each stage still needs independent implementation and tests before it is marked available.
