# AUTOBLITERATUS

AUTOBLITERATUS is an experimental Python toolkit for model ablation, evaluation, and quantization workflows. The installable application is the `autobliteratus` package under `src/autobliteratus/`; root-level Python scripts are preserved legacy/prototype files and are not the active packaged runtime.

## Source of truth

- **Current repository layout and status rules:** [docs/REPOSITORY_STATE.md](docs/REPOSITORY_STATE.md)
- **Implementation status:** [docs/implementation-status.md](docs/implementation-status.md)
- **Pipeline and optional stages:** [docs/advanced-stages.md](docs/advanced-stages.md)
- **Run plans, placement, and sharded exports:** [docs/run-plans-and-sharding.md](docs/run-plans-and-sharding.md)
- **Hardware evaluation matrix:** [benchmarks/hardware-matrix.md](benchmarks/hardware-matrix.md)
- **Dated audit and benchmark evidence:** [docs/](docs/)

AUTOBLITERATUS is meant to coordinate many tuning strategies behind a single execution model, including:

- automated ablation and post-ablation capability tuning
- architecture detection and architecture-aware tuning
- quantization scheme selection and calibration
- LoRA and fine-tuning compatibility workflows
- batch, latency, throughput, memory, and deployment tuning
- attention-head, embedding, normalization, positional encoding, and KV-cache tuning
- safety and robustness evaluation
- jailbreak resistance and prompt-injection testing
- backdoor, poisoning, leakage, and drift detection
- multilingual, multimodal, math, coding, and reasoning evaluation
- runtime feature toggles and dependency-aware orchestration
- rollback markers and reproducibility tracking
- model-card and provenance export support

## Status

The package contains 100 tuning-toggle records. Twenty-two have callable pipeline entrypoints; 78 named research modules remain unavailable prototypes. The CLI lists each status and rejects unavailable stages before loading a model. Mutating stages are opt-in and require explicit mutation authorization, a transactional checkpoint, a promoted mapping capability, and validated export. The sandbox discovery pipeline separately records bounded unknown-factor experiments and promotes only replayable, regression-safe evidence into dynamic mapping history. Do not interpret a prototype's name or class file as evidence its feature is operational.

For the current function-first audit and release-readiness assessment, see [`docs/function-first-pipeline-audit-2026-11-17.md`](docs/function-first-pipeline-audit-2026-11-17.md). Discovery evidence rules are documented in [`docs/automated-discovery-and-dynamic-mapping.md`](docs/automated-discovery-and-dynamic-mapping.md), and runtime promotion requirements are in [`docs/runtime-mapping-promotion-gate.md`](docs/runtime-mapping-promotion-gate.md).

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.40+
- Accelerate 0.30+
- PEFT 0.10+
- Safetensors 0.4+
- PyYAML 6.0+
- NumPy 1.24+
- tqdm 4.65+

The complete dependency list is maintained in [`pyproject.toml`](pyproject.toml).
Use a virtual environment; calibrated quantization requires PyTorch 2.6+ and
Optimum Quanto, and the benchmark extra adds Hugging Face Datasets:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,quantization,benchmark]"
```

On Linux, Quanto int4 may compile a C++ extension on first use. Install the
matching Python development headers and a C++ toolchain (for example, Ubuntu:
`sudo apt install python3-dev build-essential`); activate the virtual
environment so its `ninja` executable is on `PATH`.
On the tested ARM64 CPU stack, Quanto int4 also raised a tensor-version-counter
error when the caller wrapped forward inference in `torch.inference_mode()`;
use `torch.no_grad()` and validate this behavior again on the exact PyTorch and
Quanto versions used in deployment.

Registry counts are generated from the active package using `python scripts/audit_repository_state.py`. Counts printed in older dated reports are historical. A callable module-level entry point indicates registry availability only; it is not proof of production qualification or hardware support.

## Installation

Python 3.10+ is required. Install the project and development tools with:

```bash
python -m pip install -e '.[dev,quantization,benchmark]'
```

The quantization extra uses optional backend dependencies; availability and support depend on the exact model, PyTorch/Quanto versions, and hardware. See [pyproject.toml](pyproject.toml).

## Quick start

List the currently discovered toggles:

```bash
autobliteratus --list-toggles
```

<<<<<<< HEAD
Run an inspection or diagnostic pipeline on a model. Mutating stages are disabled
by default:

```bash
autobliteratus --model-path /path/to/model --work-dir ./abliterate_workspace --dry-run
```

To authorize a modifying run, the caller must explicitly pass `--allow-mutation`,
enable the mutation stage, enable validated model export, and provide a promoted
mapping capability through the runtime integration. Successful export is not
approval for deployment.

Choose a different output location or projection target:
=======
Run against a model you are authorized to modify, using a distinct output directory:
>>>>>>> origin/main

```bash
autobliteratus --model-path /path/to/model \
  --work-dir ./abliterate_workspace \
  --output-dir ./model-copy
```

Ablation intentionally changes model behavior and can weaken safety safeguards. Evaluate the resulting model independently; a successful export is not a deployment approval or a quality/safety guarantee. The source model should remain separate from the output.

Preview an example run plan without loading model weights:

Quantization is mutation-capable. The example below describes the quantization
inputs, but a modifying invocation must also pass `--allow-mutation`, explicitly
enable validated export, and supply a promoted mapping capability to the runtime
pipeline.

```bash
autobliteratus --config configs/run-plan.example.yaml --plan-only
```

## Project checks

```bash
make test
make lint
make lint-all
make build
```

<<<<<<< HEAD
The runner compares full precision, Quanto int8/int4 weight-only, and
weight-plus-activation int8/int4 controls. Generation throughput is measured
after configurable warm-up generations. It records WikiText-2 perplexity, TruthfulQA multiple-choice
accuracy, GSM8K exact match and generation tokens/second, a refusal-marker
heuristic, artifact size, and model revision. It uses small seeded samples to
keep each run bounded; these scores are directional signals, not release
acceptance thresholds or comprehensive safety/quality certification. The
runner measures only the hardware it executes on. The loader and runner support
PyTorch CPU, CUDA, MPS, XPU, HPU, or NPU placement only when that backend is
present in the installed PyTorch build; this is placement support, not a claim
that every quantization kernel supports every device. Use a separate run and
distinct output directory on each machine/accelerator; do not infer results
across hardware types. For a like-for-like hardware comparison,
pass the same `--dtype` (for example, `--dtype float32`) on each machine and
keep the model revisions, seed, datasets, and sample counts identical.
Use repeated `--calibration-momentum 0 --calibration-momentum 0.9 --calibration-momentum 1`
to compare Quanto's EMA endpoints and default; `--calibration-no-streamline`
is an additional diagnostic control. These options do not replace representative
calibration data. See [`docs/hardware-matrix.md`](docs/hardware-matrix.md)
for the device/model matrix and per-platform caveats.
=======
The test ledger is in [testing/TESTING_LOG.md](testing/TESTING_LOG.md) and [testing/results.csv](testing/results.csv). Record the commit, environment, exact command, and outcomes; do not treat an old result as a current run.
>>>>>>> origin/main

## Repository map

- `src/autobliteratus/` — canonical package source.
- `tests/` — automated tests and fixtures.
- `benchmarks/` — benchmark tools and result artifacts.
- `configs/` — example and project configuration.
- `docs/` — user/developer documentation and dated evidence.
- `ui/model-mapping-observatory/` — separate UI application.
- `legacy_root/` — reserved destination for any future fully reviewed legacy relocation; root-level prototypes are not yet represented there and must not be assumed moved.

<<<<<<< HEAD
Stages with only a class API and no pipeline-compatible module-level
`run(adapter, context)` function are prototypes and are rejected when explicitly
enabled or disabled; they are never reported as executed. The architecture
inspector reports observed structure and does not assert backend support. The
exporter records the deployment-profile metadata but does not claim profile-
specific optimization has occurred.

At this repository revision, 22 of 100 registered tuning toggles have callable
pipeline stages; the other 78 are deliberately unavailable and are not
production implementations. This is a remaining release blocker for any
workflow that depends on those capabilities; their names are not evidence that
the behavior has been implemented.

Dry-run a configuration without executing full tuning:

```bash
autobliteratus --model-path /path/to/model --dry-run
```

You can also invoke the package directly:

```bash
python -m autobliteratus --model-path /path/to/model --work-dir ./abliterate_workspace --dry-run
```

## CLI overview

The project exposes a CLI through `autobliteratus` and the module entrypoint in `src/autobliteratus/__main__.py`.

Key options include:

- `--model-path` - path or Hugging Face model ID to tune
- `--work-dir` - scratch directory; default model output is `<work-dir>/modified_model`
- `--output-dir` (`--output-path`) - new directory for the complete exported model
- `--ablation-strength` - refusal-direction projection strength, in `(0, 1]`
- `--ablation-component` - `attn.out` or `mlp.down`
- `--quantization` - `none`, calibrated Quanto int4/int8, or int4/int8 weight-only controls
- `--device` - `auto`, `cpu`, `cuda:N`, `mps`, `xpu:N`, `hpu`, or `npu` (backend must exist in the PyTorch build)
- `--config` / `--plan-only` - load a strict YAML/TOML run plan and preview the operation graph without loading weights
- `--device-map` - Accelerate placement/sharding: `auto`, `balanced`, `balanced_low_0`, or `sequential`
- `--max-memory` / `--offload-folder` - per-device memory ceilings and optional Accelerate CPU/disk offload
- `--max-shard-size` - control maximum Transformers safetensors shard size for exported weights
- `--dtype` - `auto`, `float32`, `float16`, or `bfloat16`
- `--calibration-data` - representative `.txt`, `.jsonl`, or `.json` corpus; required for activation-quantized modes
- `--calibration-samples` - maximum calibration examples (default: 128)
- `--calibration-max-length` - maximum tokens per sample (default: 256)
- `--calibration-momentum` - activation-range moving-average momentum (default: 0.9)
- `--calibration-no-streamline` - disable Quanto's default calibration streamline policy for diagnostic comparison
- `--target-frameworks` - currently only `transformers` is implemented; unsupported targets fail before loading
- `--deployment-target` - records `server`, `desktop`, `edge`, or `mobile` in export metadata; it does not optimize or certify the artifact for that profile
- `--enable` - toggle IDs to force on
- `--disable` - toggle IDs to force off
- `--list-toggles` - list registered tuning modules
- `--dry-run` - validate configuration without running the full pipeline
- `--trust-remote-code` - opt in to executing custom model-repository Python code (off by default)

See [`Run plans, acceleration, and sharded artifacts`](docs/Run%20plans%2C%20acceleration%2C%20and%20sharded%20artifacts.md) for precedence, safety,
multi-device placement, and sharded export details. Placement/sharding controls
are implemented but are not a claim of tested acceleration on every backend;
the current development host is ARM64 CPU-only.

## Repository layout

- `pyproject.toml` - project metadata and Python package configuration
- `src/autobliteratus/` - canonical package source; legacy root-level duplicates are intentionally absent
- `Makefile` - repeatable install, test, lint, benchmark-help, and build targets
- `docs/` - audits, handoff records, safety contracts, and historical evidence
- `README.md` - project overview and usage
- `LICENSE` - MIT license
- `benchmarks/results/` - retained benchmark evidence; generated model/runtime outputs are not tracked
- `tests/` - test coverage area

## Tests

With the virtual environment active and the quantization extra installed, run
the complete suite and the load-edit-export-reload matrix with:

```bash
python -m pytest -q
python -m pytest tests/test_quantization_integration.py -q
```

Equivalent repeatable build targets are available through `make test-core`,
`make test-quantization`, `make test`, `make build`, and `make benchmark-help`.
Record new validation results in the relevant dated audit or handoff report;
generated model/runtime artifacts and scratch ledgers are not tracked.

The quantization integration suite uses tiny random Llama-, Mistral-, Qwen2-,
and GPT-2-shaped models plus weight-only controls. A separate structural test
checks decoder-layer and projection-target discovery for GPT-2, Mistral, Qwen2,
OPT, GPT-NeoX, and BLOOM. These layout tests do not establish universal support
for every Transformers class, tokenizer, multimodal input, encoder-decoder model,
or quantization backend. The integration suite checks conversion, source
preservation, export, reload, and inference plumbing only. Randomly initialized
weights cannot measure real-world accuracy, hallucinations, safety behavior, or
reasoning. Architecture placement and quantization-kernel coverage differ; a
model loading successfully on a device does not imply that a selected backend
can quantize it or run it efficiently. Each device/backend combination needs a
separate measured run and result record.

## Example workflow

1. Select a target model.
2. Choose a quantization and deployment profile.
3. Enable or disable specific tuning toggles.
4. Run `autobliteratus`.
5. Review generated artifacts, provenance, and reproducibility markers.

## Important note

This toolkit performs model transformation, optimization, and evaluation
workflows. It should only be used on models and datasets that you are
authorized to process, and it should be run in a controlled environment with
proper validation. A successful export means the configuration, tokenizer,
and weight-file structure were validated; it is not a benchmark or a guarantee
that downstream task quality has been preserved. Review and evaluate the
result before deployment. The source model is loaded and edited only in
memory; the package writes the resulting model to the separate output path.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
=======
See [docs/REPOSITORY_STATE.md](docs/REPOSITORY_STATE.md) for audited paths, known legacy naming discrepancies, and the relocation policy.
>>>>>>> origin/main
