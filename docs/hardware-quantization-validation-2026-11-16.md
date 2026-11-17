# Hardware Capability and Quantization Validation — 2026-11-16

## Result summary

- Baseline active stages before Phase B: 18; all 18 were registry/contract validated.
- Phase B latency benchmark promoted during this run: 1 additional active stage; current registry: 19 active stages.
- Callable registry entries: 19.
- Missing callable entries: 0.
- Host architecture: aarch64.
- PyTorch: 2.15.0+cu132.
- CPU: available.
- CUDA: False.
- MPS: False.
- XPU/HPU/NPU: unavailable on this runner.
- Accelerate: installed.
- Optimum Quanto: unavailable.

## Active stage registry validation

| Stage | Callable entrypoint | Hardware-independent contract |
|---|---:|---:|
| `tuning.anti_router_collapse` | yes | imported and exposed by `build_registry()` |
| `tuning.architecture_detection` | yes | imported and exposed by `build_registry()` |
| `tuning.audit_trail_auto` | yes | imported and exposed by `build_registry()` |
| `tuning.auto_ablated_capability` | yes | imported and exposed by `build_registry()` |
| `tuning.auto_security_scanning` | yes | imported and exposed by `build_registry()` |
| `tuning.baseline_auto_pre_post` | yes | imported and exposed by `build_registry()` |
| `tuning.benchmark_regression_auto` | yes | imported and exposed by `build_registry()` |
| `tuning.calibrated_quantization` | yes | imported and exposed by `build_registry()` |
| `tuning.latency_auto` | yes | imported and exposed by `build_registry()` |
| `tuning.memory_footprint_auto` | yes | imported and exposed by `build_registry()` |
| `tuning.model_card_generation_auto` | yes | imported and exposed by `build_registry()` |
| `tuning.model_export` | yes | imported and exposed by `build_registry()` |
| `tuning.multi_objective_auto` | yes | imported and exposed by `build_registry()` |
| `tuning.provenance_recording_auto` | yes | imported and exposed by `build_registry()` |
| `tuning.refusal_direction_ablation` | yes | imported and exposed by `build_registry()` |
| `tuning.reproducibility_auto` | yes | imported and exposed by `build_registry()` |
| `tuning.tokenization_consistency_auto` | yes | imported and exposed by `build_registry()` |
| `tuning.version_compatibility_auto` | yes | imported and exposed by `build_registry()` |
| `tuning.weight_fingerprint_auto` | yes | imported and exposed by `build_registry()` |

## Quantization benchmark execution

The benchmark CLI was exercised with `--help`; it exposes pinned `--revision MODEL=REVISION`, device, dtype, calibration, dataset sample, and quantization-mode controls. Full model/dataset execution was not falsely reported as completed because Optimum Quanto is not installed on this runner and CUDA/XPU are unavailable.

CUDA probe exit=Requested 'cuda', but the cuda backend is unavailable.
XPU probe exit=Requested 'xpu', but the xpu backend is unavailable.

## Interpretation

The executed evidence qualifies registry wiring, stage importability, CPU runtime availability, and the benchmark CLI contract. It does not qualify CUDA, XPU, mobile power, or full quantization quality. Those paths remain environment-gated and require a runner with the target accelerator plus the `quantization` extra. The benchmark is comparative evidence only; it is not a safety or deployment guarantee.
