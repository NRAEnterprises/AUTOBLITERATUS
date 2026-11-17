# Memory and Weight Fingerprinting Hardening Coverage

## Scope

Focused contract probes were run against:

- `src/autobliteratus/tuning/_44_memory_footprint_auto_tuning.py`
- `src/autobliteratus/tuning/_66_weight_fingerprint_auto_tuning.py`

The probe used a deterministic tiny `torch.nn.Parameter` model stub with `hf_device_map`, `parameters()`, and `state_dict()` support. It exercised the direct stage contracts rather than relying on the existing suite, which does not currently import these two stages.

## Results

- Contract probe: **passed**.
- Memory stage: **49% statement coverage**, 20 of 39 statements covered.
- Weight fingerprint stage: **97% statement coverage**, 35 of 36 statements covered.
- HTML report: `htmlcov-hardening/index.html`.

### Memory telemetry

Observed report shape:

```text
measurement_point: after selected transformations, before export
host_rss_bytes: 577314816  # environment-local measurement
accelerator_allocated_bytes: {}
optimization_performed: false
```

The stage correctly records host RSS and reports no optimization. It currently exercises accelerator paths only when the model advertises non-CPU/non-meta devices and the corresponding PyTorch backend exposes a compatible allocation API.

Uncovered memory lines are the fallback and backend-specific branches:

- `/proc/self/status` failure fallback to `psutil`.
- Missing `psutil` / OS failure path.
- Per-device CUDA/ MPS/ other accelerator allocation calls.
- Runtime/type/value error handling around allocation APIs.

### Weight fingerprinting

Observed report shape:

```text
sha256: 64 hexadecimal characters
tensor_count: 1
stage: final-pre-export
```

The bounded-memory SHA-256 path, provenance helper integration, enabled/disabled behavior, file persistence, and verification path passed. One branch remains uncovered: verification when the expected fingerprint file is absent while fingerprinting is enabled.

## Hardening requirements

1. Add direct unit tests for `/proc` failure and `psutil` fallback.
2. Add backend-mocked tests for CUDA allocation, MPS `current_allocated_memory`, unsupported devices, and allocation exceptions.
3. Add tests for models with multiple device-map entries and `meta` parameters.
4. Add a negative fingerprint test proving a changed tensor fails verification.
5. Add a missing-artifact test proving enabled verification returns `False` without creating a new fingerprint unexpectedly.
6. Make fingerprint serialization include explicit dtype, shape, parameter name, and tensor count semantics; define whether ordering is part of the compatibility contract.
7. Add atomic fingerprint writes and a checksum manifest binding each fingerprint to the run ID, model source identity, and stage order.
8. Add a large-model smoke test that asserts bounded host-memory growth rather than only correctness on a tiny model.
9. Add coverage gates of at least 85% for both promoted stages; memory telemetry currently falls materially below that threshold.
