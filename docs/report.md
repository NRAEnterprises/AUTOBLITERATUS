# Partial quantization isolation matrix — ARM64 CPU

**Status: intentionally stopped after 9 successful runs; this is a partial matrix, not a complete release qualification.** The 135M model has baseline and all four controls; the 360M model has baseline, int8 weight-only, int8 weight+activation, and int4 weight-only. The remaining 360M int4 activation case and other model families were moved to separate runs to bound the slow CPU workload.

## Hardware and method

- Hardware: ARM64 Linux, 6-core ARM Neoverse-N1; no accelerator was available in this environment. This says nothing about the user’s other hardware.
- Runtime: float32, six PyTorch threads. Dataset seed 2026; calibration used 128 samples from WikiText-2 train, max sequence length 256, Quanto EMA momentum 0.9, streamline enabled.
- Evaluation per model: 16 WikiText-2 test samples, 16 TruthfulQA-MC rows, 8 GSM8K test rows, and 8 refusal-marker prompts; 64 max new tokens, two generation warmups.
- This report’s metrics are small-sample indicators: WikiText perplexity measures LM fit; TruthfulQA-MC is a factuality proxy, GSM8K exact-match is a reasoning proxy, and refusal markers are only a heuristic. They do not measure hallucination rates or certify quality/safety.

## Paired results

| Model | Mode | WikiText PPL | PPL / baseline | TruthfulQA-MC | GSM8K exact | GSM8K tok/s | tok/s / baseline | Checkpoint MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HuggingFaceTB/SmolLM2-135M-Instruct | none | 41.027 | 1.000x | 0.250 | 0.000 | 31.23 | 1.000x | 259.8 |
| HuggingFaceTB/SmolLM2-135M-Instruct | quanto-int8-weight-only | 41.206 | 1.004x | 0.250 | 0.000 | 15.28 | 0.489x | 213.5 |
| HuggingFaceTB/SmolLM2-135M-Instruct | quanto-int8-mom-0.9 | 202.184 | 4.928x | 0.188 | 0.000 | 5.59 | 0.179x | 213.5 |
| HuggingFaceTB/SmolLM2-135M-Instruct | quanto-int4-weight-only | 63.979 | 1.559x | 0.188 | 0.000 | 10.37 | 0.332x | 170.2 |
| HuggingFaceTB/SmolLM2-135M-Instruct | quanto-int4-mom-0.9 | 354.121 | 8.631x | 0.188 | 0.000 | 4.87 | 0.156x | 170.2 |
| HuggingFaceTB/SmolLM2-360M-Instruct | none | 30.502 | 1.000x | 0.250 | 0.000 | 17.26 | 1.000x | 693.4 |
| HuggingFaceTB/SmolLM2-360M-Instruct | quanto-int8-weight-only | 30.703 | 1.007x | 0.250 | 0.000 | 8.39 | 0.487x | 484.8 |
| HuggingFaceTB/SmolLM2-360M-Instruct | quanto-int8-mom-0.9 | 274.365 | 8.995x | 0.250 | 0.000 | 3.55 | 0.206x | 484.8 |
| HuggingFaceTB/SmolLM2-360M-Instruct | quanto-int4-weight-only | 43.255 | 1.418x | 0.250 | 0.000 | 5.57 | 0.323x | 357.2 |

## Interpretation

For SmolLM2-135M, int8 weight-only was 1.00x baseline perplexity but 0.49x throughput; int8 with qint8 activations was 4.93x perplexity and 0.18x throughput. Int4 weight-only was 1.56x perplexity and 0.33x throughput; int4 with activations was 8.63x perplexity and 0.16x throughput. For SmolLM2-360M, int8 weight-only was 1.007x perplexity and 0.487x throughput; int8 with activations was 8.995x perplexity and 0.206x throughput; int4 weight-only was 1.418x perplexity and 0.323x throughput.

The paired controls strongly implicate activation quantization as a major contributor to the perplexity increase: weight-only int8 stays close to baseline on both sizes while the combined mode deteriorates sharply. Throughput falls even with weight-only quantization, suggesting backend/runtime overhead or non-optimized ARM CPU kernels; this is an inference from paired timings, not a CPU operator-profile proof. The one-row-per-EMA-update calibration at momentum 0.9 is order-sensitive and remains a factor to test, not a confirmed sole cause.

## Reproducibility and limitations

- Raw per-run rows: `results.jsonl` (contains pinned Hub commit IDs, actual quantization module counts, dataset/configuration details, and metrics).
- Calibration corpus: `calibration-wikitext2-train-seed-2026.jsonl`; SHA-256: `1ebe245c8e347a1a95a847d9fe617f76b2f007850f1ac82c48dbd116249e629c`.
- All 9 completed rows succeeded, but there are only two checkpoints and both are Llama-family; broader model-base evaluation is in `../arm64-weight-only-family-matrix/`.
- Only Neoverse-N1 CPU was measured here. No conclusions are drawn about x86, CUDA/ROCm, MPS, XPU, HPU, NPU, or any devices the user may own.
- Any refusal ablation was disabled for this quantization matrix; these comparisons do not certify safety or logic/reasoning behavior.
