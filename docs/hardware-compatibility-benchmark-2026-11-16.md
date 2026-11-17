# Hardware Compatibility Benchmark Evidence

## Execution result

A hardware compatibility suite was executed against the available environment and declared targets.

### Available evidence

- Host architecture: `aarch64`
- CPU: 6-core Neoverse-N1
- CUDA runtime: PyTorch reports CUDA 13.2 support, but `cuda_available=False`
- NVIDIA driver/tooling: `nvidia-smi` unavailable
- CUDA device count: `0`
- Benchmark CLI wiring: passed
- Integration target: exit 0, 17 tests skipped by environment guards
- Quantization target: 4 passed, 17 skipped
- CPU/ARM64 host is the only hardware path actually exercised in this sandbox.

### Not executed

- x86-64 CPU benchmark: unavailable because this environment is ARM64 and no x86-64 runner is attached.
- CUDA benchmark: unavailable because no NVIDIA device, driver, or `nvidia-smi` is present.
- MPS/ROCm/XPU/HPU/NPU: unavailable in this environment.

No CUDA or x86-64 pass claim is made. The correct next step is CI or attached-runner execution, not emulation or extrapolation.

## Benchmark commands

```text
PYTHONPATH=src .venv/bin/python benchmarks/quantization_quality_matrix.py --help
PYTHON=./.venv/bin/python PYTEST='./.venv/bin/python -m pytest' make test-integration
PYTHON=./.venv/bin/python PYTEST='./.venv/bin/python -m pytest' make test-quantization
```

## Interpretation

The existing integration tests are environment-gated and therefore provide compatibility wiring evidence, not cross-hardware qualification. Add a required matrix with separate runners for ARM64 CPU, x86-64 CPU, and CUDA. Persist model revision, PyTorch/Transformers/Accelerate/Quanto versions, dtype, thread count, device name, latency/throughput, finite-metric status, and artifact checksums for each row.
