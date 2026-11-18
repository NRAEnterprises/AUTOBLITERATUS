# Run plans, placement, and sharded artifacts

The CLI accepts strict YAML or TOML run plans. Configuration parsing and precedence are implemented in `src/autobliteratus/run_config.py` and `src/autobliteratus/cli.py`; YAML uses `safe_load`, unknown keys are rejected, and explicit CLI values override values from the plan.

Preview the checked-in example without downloading model weights:

```bash
autobliteratus --config configs/run-plan.example.yaml --plan-only
```

The example is intentionally plan-only and uses a placeholder model identifier. Replace it with a model you are authorized to use before execution.

```yaml
model_path: /path/to/authorized/model
work_dir: ./abliterate_workspace
output_dir: ./model-copy
quantization: none
device: cpu
device_map: null
target_frameworks:
  - transformers
deployment_target: server
```

Placement options such as `device-map`, memory limits, and offload are available in the CLI, but they do not prove accelerator performance or universal quantization support. Consult `docs/hardware-matrix.md` for the evaluation matrix and its limits. Export paths must remain separate from source model paths; inspect the exporter and test suite for exact safety checks.

For the detailed historical hardware, sharding, and run workflow notes, see `Run plans, acceleration, and sharded artifacts.md` in this directory. This canonical page supplements that record and does not claim that every backend listed there has been qualified.
