# Opt-in diagnostic and selection stages

Six formerly unavailable stages now expose the pipeline contract `run(adapter, context)`. They default off to avoid unexpected model passes, expensive scans, or implicit model edits. Enable them explicitly with CLI toggles or a YAML/TOML profile. All reports are surfaced in the normal pipeline result.

## Router utilization diagnostics

Enable `tuning.anti_router_collapse` on a mixture-of-experts model. Supply representative `routing_probe_texts` in `stage_options`, or reuse the already validated `calibration_data` text file. The stage attaches bounded forward hooks to modules named `router`, `gate`, or `router_gate`, runs at most `routing_max_samples` texts truncated to `routing_max_length`, and computes average expert load and normalized Shannon entropy per observed router. `routing_entropy_threshold` is in `[0, 1]`; layers below it are flagged as low-utilization diagnostics.

The stage is read-only. It does **not** rebalance experts, retrain the router, establish a backdoor, or claim that low entropy is necessarily a defect. If the model is dense, no probe text is provided, or the router layout is unrecognized, the report gives an explicit non-applicable/not-evaluated status instead of reporting a successful measurement.

## Multi-objective candidate selection

Enable `tuning.multi_objective_auto` and supply `multi_objective_candidates`, each containing an `id` (optional), a `config` mapping, and finite numeric `scores`. `multi_objective_objectives` maps score names to `max` or `min`; `multi_objective_weights` sets non-negative scalarization weights. The stage validates input, removes Pareto-dominated candidates, min-max normalizes the frontier, applies directions and weights, then selects the highest-scoring deterministic compromise (first candidate wins exact ties).

This is **selection only**: candidate generation, model execution, measurement, and candidate artifact export are not performed by this stage. A missing candidate list returns `no_candidates`; missing/non-finite objective values fail with an actionable error.

## Numeric parameter scan

Enable `tuning.auto_security_scanning` to scan materialized floating-point parameters. `security_scan_max_abs` defaults to `1000.0`; larger absolute values produce heuristic warnings, while NaN/infinity values produce critical numeric-integrity findings. Non-floating and meta-device parameters are counted separately. The stage is read-only.

This limited check is **not** a malware, trojan, backdoor, poisoned-data, adversarial-behavior, or code-security detector. A clean report means only that these numeric checks did not identify an anomaly; a magnitude warning may be expected for some model designs.

## Paired baseline and benchmark regression gate

Enable `tuning.baseline_auto_pre_post` to measure the loaded model before any configured ablation or quantization. Enable `tuning.benchmark_regression_auto` to measure it again after the enabled transformations. Selecting the regression stage automatically enables the baseline stage; explicitly disabling the baseline while requesting regression is rejected. The pipeline orders baseline capture before model edits and the regression gate after them. A missing or failed baseline prevents the regression gate and export from being treated as successful.

Provide exactly one of `stage_options.evaluation_texts` (a list of strings) or `stage_options.evaluation_data` (a local `.txt`, JSON, or JSONL file containing strings or `text`/`prompt` records). Evaluation data is required; calibration text is not silently reused. The defaults cap evaluation at 64 samples and 256 tokens per sample; override them with `evaluation_max_samples` and `evaluation_max_length`. Text is kept in memory for measurement and is not written into the audit files.

The built-in battery reports token-weighted `mean_nll` and capped `perplexity` (lower is better) plus `forward_tokens_per_second` (higher is better). This measures causal-LM next-token fit and model-forward throughput—not text-generation throughput, hallucinations, reasoning, safety, or deployment quality. Non-finite outputs, missing metrics, mismatched metric names, and unscored samples fail closed. Per-metric tolerance ratios are configurable with `regression_tolerances`; defaults are 2% for NLL/perplexity and 10% for throughput because single-run timing is noisier. For example:

```yaml
enable:
  - tuning.benchmark_regression_auto
stage_options:
  evaluation_texts:
    - "A representative evaluation sentence, not calibration/training text."
    - "A second representative evaluation sentence."
  evaluation_max_samples: 2
  evaluation_max_length: 128
  regression_metric_directions:
    mean_nll: min
    perplexity: min
    forward_tokens_per_second: max
  regression_tolerances:
    mean_nll: 0.02
    perplexity: 0.02
    forward_tokens_per_second: 0.10
```

If any valid metric exceeds its tolerance, the stage records its comparison and fails the pipeline; the export stage is skipped. Baseline capture and regression execution are opt-in to avoid additional model passes by default.

Preview a local evaluation-gate YAML plan’s ordering and validation without downloading model weights with `PYTHONPATH=src python -m autobliteratus --config <path-to-plan.yaml> --plan-only`.

## Run audit

Enable `tuning.audit_trail_auto` to write a unique JSONL and Markdown audit under `<work_dir>/audit_trails/` after orchestration finishes. Each file is written atomically and includes stage executed/skipped/failed status, safe run settings, runtime versions, artifact validation summary, and available evaluation summaries. The source model reference is represented by a hash. Raw exception messages, arbitrary stage options, prompts, and evaluation text are omitted. The audit directory is checked against the source and export paths; an audit write failure marks the selected audit stage failed.

## Example profile

```yaml
model_path: /models/moe-model
work_dir: ./run-work
output_dir: ./moe-model-copy
# Use routing diagnostics only for a model with a compatible MoE router.
enable:
  - tuning.anti_router_collapse
  - tuning.multi_objective_auto
  - tuning.auto_security_scanning
stage_options:
  routing_probe_texts:
    - "A representative domain-specific prompt."
  routing_max_samples: 1
  routing_max_length: 128
  routing_entropy_threshold: 0.5
  security_scan_max_abs: 1000.0
  multi_objective_objectives:
    quality: max
    latency_ms: min
  multi_objective_weights:
    quality: 0.7
    latency_ms: 0.3
  multi_objective_candidates:
    - id: candidate-a
      config: {quantization: none}
      scores: {quality: 0.91, latency_ms: 100}
    - id: candidate-b
      config: {quantization: int8}
      scores: {quality: 0.89, latency_ms: 60}
```

Candidate scores in this example are illustrative inputs only; the toolkit does not create or validate them. Replace them with measured scores from an evaluation process. A profile can be inspected without loading model weights using `autobliteratus --config profile.yaml --plan-only`; the actual stages execute only when the plan runs on a model.
