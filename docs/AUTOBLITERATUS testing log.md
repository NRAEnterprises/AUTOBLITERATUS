# AUTOBLITERATUS testing log

> Historical testing log retained for traceability. The current repository no longer tracks the append-only ledger paths mentioned below. Current validation is recorded in the dated audit and handoff reports.

This is the historical edit/fine-tuning ledger. It is retained for traceability; the former append-only ledger files are not part of the cleaned runtime repository. Current validation belongs in `docs/HANDOFF_PRIORITY3_2026-11-17.md` and the current function-first audit. A result marked `invalid` is retained for audit but must not be used as a successful metric.

## Record format

`results.csv` columns: `run_id,date,status,scope,hardware,command,passed,failed,metrics,artifact,notes`.

Use these statuses consistently:

- `passed`: automated validation completed successfully.
- `measured`: benchmark row has finite/valid metrics; use `notes` to specify scope and limitations.
- `failed`: operation/test failed; include the failure cause and retain logs.
- `invalid`: execution completed but metric is unusable (for example, NaN logits or a known defective reloader).
- `not-run`: target unavailable or not attempted; never treat as a pass.

## Historical baseline records

The latest automated suite result before the config/plan/sharding work was 149 passing tests on the ARM64 sandbox. It is a historical baseline; rerun after each implementation batch and add a new row instead of changing this entry.

| Run | Result | Details |
|---|---|---|
| `full-pytest-pre-plan-config` | 149 passed | `pytest -q`; ARM64 CPU, Python 3.12, local project venv. Included the GPT-NeoX legacy output-head reload regression. |
| `arm64-family-matrix` | 5 valid, 2 invalid, 2 failed | Three model families and full-precision/int8/int4 weight-only controls; see linked report and raw results. |
| `arm64-pythia-compatible-rerun` | 3 valid rows | Corrected Pythia reload path; int8 was close in PPL but slower, int4 had a notable quality and throughput cost. |

## Benchmark interpretation caveats

The recorded task subsets (8 WikiText samples; four TruthfulQA-MC, GSM8K, and refusal-marker samples) are comparative smoke probes, not broad quality, safety, reasoning, or hallucination evaluations. Throughput is same-run and CPU timing varies. A single ARM64 CPU host does not establish performance on x86-64, CUDA/ROCm, MPS, XPU, HPU, or NPU. Keep model revisions, dataset splits, seeds, thread counts, dtype, context/generation lengths, and warmups paired between comparisons.

## Current implementation batch — 2026-11-16

The first full suite after config, plan-only, CPU device-map, and sharded-export work passed **157 tests** on the ARM64 CPU sandbox. After the tied-head reload correction, the final suite passed **158 tests** in 9.94 seconds. The sample YAML plan-only command, tiny local GPT-2 Accelerate `device_map=auto` CPU load, and tiny Quanto sharded export/reload case passed. Focused Ruff checks now cover the run-plan loader and quantization compatibility code/tests. Wheel and source distribution builds passed after installing the declared `build` dev extra, and benchmark `--help` exited successfully without downloading models.

The latest full active-code Ruff target remains **failed** with 184 findings (170 auto-fixable) across source, tests, and benchmarks; an earlier scan reported 187. This is recorded as a project-wide lint backlog, not hidden by the focused pass. The first build/lint invocations also failed because `build` and `ruff` were missing from the inherited virtual environment; see [`environment-setup-findings.md`](environment-setup-findings.md) for the failure/resolution sequence.

Exact historical commands and outcomes remain in this dated document. No CUDA, multi-GPU, ROCm, MPS, XPU, HPU, or NPU run occurred; CPU dispatch and a shard/reload path are not substitutes for those hardware validations.


## Larger-model tied-head regression — 2026-11-16

A first Qwen2.5-1.5B-Instruct ARM64 CPU matrix exposed that the compatibility loader rejected a missing `lm_head.weight` even though the model configuration ties it to a present, same-shaped input embedding and the head itself is not quantized. The initial int8/int4 rows remain recorded as **failed**; no quality or throughput metrics are attributed to them. The loader now tolerates only this verified tied-output alias, while still failing on missing embeddings, shape mismatch, quantized heads, other missing keys, or unexpected keys. The focused tied-weight regression and final full suite both passed; the full suite result is 158 tests in 9.94 seconds.

The matched post-fix run used the same pinned Qwen revision, seed 2026, float32, six ARM64 CPU threads, 8 WikiText samples and 4 examples each for TruthfulQA-MC, GSM8K, and refusal probes. All three rows reloaded and produced finite metrics. Relative to the paired full-precision baseline (PPL 19.206; 3.874 generated tokens/s), int8 measured PPL 19.404 (1.010x) and 2.136 tokens/s (0.551x); int4 measured PPL 24.866 (1.295x) and 1.434 tokens/s (0.370x). These small probes show smaller checkpoints but no CPU speedup, with a substantial int4 quality/throughput penalty. They do not establish hallucination, safety, reasoning, or deployment quality. The original raw benchmark directories are not retained in the cleaned repository.


## Three callable stage promotions — 2026-11-16

Promoted three previously unavailable entries as opt-in (`default: false`) pipeline stages. `tuning.anti_router_collapse` now probes observed expert-load entropy from bounded representative MoE prompts and is strictly diagnostic; it does not rebalance weights. `tuning.multi_objective_auto` validates externally scored candidates, computes a direction-aware Pareto frontier, and picks a weighted normalized compromise; it does not generate/evaluate candidates. `tuning.auto_security_scanning` scans materialized float parameters for non-finite values and configurable extreme magnitudes in bounded chunks; it is not malware/backdoor detection. The CLI and YAML/TOML `stage_options` path report results and validate allowed option names.

At that historical snapshot the registry reported **12 callable stages and 87 unavailable prototypes out of 99**. The current registry reports 100 total, 22 callable, and 78 unavailable; see `PROJECT_HANDOFF_MANIFEST.yaml`. The historical focused validation and hardware limitations remain useful only as dated evidence.
