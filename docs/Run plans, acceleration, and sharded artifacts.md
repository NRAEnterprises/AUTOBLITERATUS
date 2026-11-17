# Run plans, acceleration, and sharded artifacts

## Run plans

`--config` loads a safe YAML (`.yaml`/`.yml`) or TOML (`.toml`) mapping. Unknown keys are rejected; YAML uses `safe_load`, not executable YAML constructors. An optional `[autobliteratus]` TOML table or `autobliteratus:` YAML section is accepted. CLI arguments explicitly supplied on the command line override their matching config values; list options supplied on the CLI replace the corresponding profile list.

```bash
autobliteratus --config configs/run-plan.example.yaml --plan-only
```

`--plan-only` validates the configuration and dependency graph, lists planned/skipped/unavailable stages, resolves the output target, and exits before model weights are loaded. It is distinct from `--dry-run`, which loads and inspects the model while suppressing modifications and export. A plan containing a requested prototype is reported as unsuccessful with that stage under `unavailable`; regular execution remains fail-closed.

Config keys correspond to CLI destinations such as `model_path`, `output_dir`, `quantization`, `enable`, `disable`, `device_map`, `max_memory`, and `max_shard_size`. See [`configs/run-plan.example.yaml`](../configs/run-plan.example.yaml). `trust_remote_code` remains false by default; set it only for a model whose code you have reviewed.

## Acceleration and placement

The CLI still accepts an explicit `--device` (`cpu`, `cuda:N`, `mps`, `xpu:N`, `hpu`, `npu`, or `auto`). `--device-map` opts into Transformers/Accelerate dispatch and module sharding:

```bash
autobliteratus \
  --model-path /models/source \
  --output-dir /models/tuned-copy \
  --device-map auto \
  --max-memory 0=20GiB \
  --max-memory cpu=48GiB \
  --offload-folder /scratch/autobliteratus-offload \
  --max-shard-size 2GB
```

Supported placement strategies are `auto`, `balanced`, `balanced_low_0`, and `sequential`. When a memory budget or offload folder is supplied without a map, the pipeline selects `auto`. An offload folder is scratch space, not the final artifact directory, and must not overlap the source or output. `device_map` dispatch requires Accelerate and a model supported by the installed Transformers version. A successful load/dispatch is not evidence that every quantization kernel is accelerated on that backend; validate conversion, inference, and save/reload on the target machine.

`--max-shard-size` controls the Transformers safetensors export limit. Quantized Quanto wrappers are saved through their underlying Transformers model because Quanto 0.2.7's public serializer drops the shard-size argument. The exporter then writes the Quanto quantization map and validates the artifact. Tiny tensors can exceed the per-shard cap individually; Transformers keeps an individual tensor intact and may put it in a shard larger than the configured limit.

## Run, log, and validate

```bash
# Preview the stage graph without downloading a remote model.
python -m autobliteratus --config configs/run-plan.example.yaml --plan-only

# Run the project checks.
make test-core
make test-quantization
make test
make build

# Or install all development/test extras once.
make install-dev
```

The output includes a SHA-256 checksum map for files present before the manifest is written, runtime/package information, device-map/memory settings, and the configured shard size. Check `testing/TESTING_LOG.md` and append one new row to `testing/results.csv` for every test or benchmark run. Keep `failed`, `invalid`, and `not-run` outcomes; do not replace them with later successful runs.

The current development host is ARM64 CPU-only. CPU config/plan logic and tiny model sharding/reload are covered with tests, but actual multi-GPU, CUDA, ROCm, MPS, XPU, HPU, and NPU placement/offload are not claimed as tested here. Those targets need paired end-to-end runs and log entries on compatible hardware before being advertised as validated.
