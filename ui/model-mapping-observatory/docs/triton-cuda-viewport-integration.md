# Triton/CUDA Viewport Integration Notes

## Verified current boundary

The Model Mapping Observatory uses Three.js/WebGL for the browser-side 3D graph, matrix surface, and mind-map framing. The browser receives bounded mapping metadata, dimensions, evidence status, fingerprints, and measured-summary fields. It does not receive unrestricted model weights and it does not execute Triton or CUDA kernels.

The observatory's `Triton / CUDA / Vulkan` card is therefore an explicit **context** indicator, not a claim of live GPU execution. The topology export preserves this distinction in `backend_context` and marks direct weight mutation as false.

## AUTOBLITERATUS telemetry path

The AUTOBLITERATUS runtime currently integrates accelerator profiling through Python/PyTorch contracts:

1. `loader.py` resolves `auto` to CUDA only when `torch.cuda.is_available()` is true, binds a concrete `cuda:0` device, and rejects unavailable or unsupported backends.
2. `_44_memory_footprint_auto_tuning.py` reports host RSS and per-device `memory_allocated` readings as observational telemetry. Unavailable readings are emitted as `null`; the stage does not optimize or mutate the model.
3. `_41_kv_cache_auto_tuning.py` measures bounded cache/sequence candidates and records accelerator allocation before and after each candidate when a backend exposes the required API.
4. `resource_budget_gate.py` correlates accelerator allocation with host RSS, latency, throughput, batch size, and sequence length. It emits report-only violations or incomplete measurements rather than selecting a deployment policy.
5. `configs/polyglot-runtime-pairings.yaml` keeps CUDA calibration environment-blocked and requires seed replay, CPU reference matching, device metadata, precision sweeps, and resource-budget gates. Triton is present in the environment package set, but there is no promoted Triton adapter entry in the current manifest.

## Promotion boundary

The polyglot contract requires disabled-by-default sidecar or versioned-extension boundaries, model/toolchain/device/input/schema fingerprints, bounded execution, structured failures, reference-equivalence tests, reproducibility, rollback, and explicit named-pipeline enablement. The Three.js viewport is not a promotion mechanism; it is an inspection and proposal surface for those evidence artifacts.
