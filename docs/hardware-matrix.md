# Hardware and architecture test matrix

The benchmark records only the hardware on which it actually runs. It does not infer the user's device inventory, and a supported PyTorch placement type is not a claim that Quanto has an optimized kernel for it. Run each cell on an actual compatible machine/build and retain its complete `manifest.json`, `results.jsonl`, `results.csv`, and `report.md`.

## Test strata

| Stratum | Example target to identify in the manifest | Placement selector | Important qualification |
|---|---|---|---|
| x86-64 CPU | Exact CPU model and ISA (baseline/AVX2/AVX-512 where applicable), OS/kernel, thread count | `cpu` | Separate ISA tiers; generic CPU placement does not guarantee a vectorized quantized kernel. |
| AArch64/ARM64 CPU | Exact SoC/CPU, ARM ISA features, OS/kernel, thread count | `cpu` | Distinguish server ARM from Apple ARM; do not transfer performance conclusions between them. |
| NVIDIA GPU | Exact GPU model/compute capability, driver, CUDA, PyTorch CUDA runtime | `cuda:0` or `auto` | Quanto's documented accelerated matrix combinations are CUDA-specific; test each precision/device pairing. |
| AMD GPU | Exact GPU, ROCm/HIP and PyTorch HIP build | `cuda:0` (PyTorch ROCm uses the CUDA device API name) | Record backend as ROCm, not NVIDIA CUDA. Successful placement is not evidence of Quanto kernel acceleration. |
| Apple Silicon | Exact SoC, macOS, PyTorch MPS availability | `mps` | Test MPS placement separately from ARM CPU; quantization op coverage can differ. |
| Intel accelerator | Exact XPU model/driver/PyTorch XPU build | `xpu:0` | Treat Quanto conversion/runtime coverage as an empirical test, not an assumed guarantee. |
| Habana Gaudi | Exact generation, SynapseAI and HPU-enabled PyTorch | `hpu` | Requires the vendor software stack and corresponding PyTorch integration. |
| Huawei Ascend | Exact NPU, CANN and torch-npu version | `npu` | Requires the vendor software stack and corresponding PyTorch integration. |

This list is an evaluation matrix, not an exhaustive claim that every listed platform has a working Quanto kernel. Add other Torch-supported device backends only after implementing and testing their explicit resolver, synchronization, model loading, quantization, save/reload, and inference paths.

## Accelerate placement and sharded artifacts

Model loading supports Transformers/Accelerate device-map strategies `auto`, `balanced`, `balanced_low_0`, and `sequential`, with repeatable per-device memory limits and an optional offload folder. This enables layer placement across multiple available devices when the model/backend combination supports it. Export sharding is separately controlled by `--max-shard-size`; it does not change load-time placement. See [`Run plans, acceleration, and sharded artifacts`](Run%20plans%2C%20acceleration%2C%20and%20sharded%20artifacts.md) for config and CLI examples. Record the resolved `hf_device_map`, backend versions, memory budgets, and offload settings for each multi-device run, then validate conversion, inference, and exported reload. No multi-GPU or accelerator result should be inferred from CPU-only tests.

## Model/family and quantization coverage

The default benchmark set spans Pythia 70M, SmolLM2 135M and 360M, GPT-2 124M, and Qwen2.5 0.5B and 1.5B. Add model IDs with repeated `--model`; pin immutable Hub commits with repeated `--revision MODEL=COMMIT`. Compare full precision with int8/int4 weight-only controls and int8/int4 weights plus qint8 activations. Repeat `--calibration-momentum` to test scale-EMA sensitivity (0 means last-batch scale; 1 means first non-default batch scale after initialization). The same calibration and evaluation sample sets, seed, dtype, sequence length, and evaluation splits should be used for paired runs.

Models with unsupported module types must produce an explicit failed row rather than a false success. For example, the pinned Quanto implementation tested here does not expose GPT-2 Conv1D weights to its weight-only conversion map; activation-enabled GPT-2 variants may behave differently and must not be assumed to quantize its weights. Check the per-run quantization report and exported qmap before interpreting a result.

## Example commands

On each machine, use a fresh output directory and preserve the same model/revision and task settings. A CPU run with a calibration sweep can be launched as follows:

```bash
python benchmarks/quantization_quality_matrix.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --model EleutherAI/pythia-70m-deduped \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cpu --dtype float32 --threads 6 \
  --calibration-samples 128 --max-length 128 \
  --calibration-momentum 0 --calibration-momentum 0.9 --calibration-momentum 1 \
  --output-dir ./benchmark-results/run-arm64-cpu
```

Run CUDA with the same parameters, a fresh output directory, and the same explicit dtype to compare the same numeric precision. Use the platform's compatible PyTorch build for MPS, XPU, HPU, or NPU. A run rejected by a backend is useful evidence and must remain in the results; do not silently substitute another device or quantization backend.

## What the metrics do not prove

The runner reports WikiText perplexity, TruthfulQA multiple-choice accuracy, GSM8K exact match, a fixed refusal-marker heuristic, and warmed-up short generation throughput. These are small comparative probes, not comprehensive measures of hallucination, reasoning, safety, calibration quality, or deployment latency. Run larger held-out sets, longer contexts and batch sizes, domain-specific samples, and operator-level profiling before release decisions. Separate model-load/conversion time from steady-state inference and separately measure prefill and autoregressive decode if throughput is a release requirement.
