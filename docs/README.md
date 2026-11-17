# AUTOBLITERATUS

AUTOBLITERATUS is an experimental Python toolkit for automated model ablation, capability tuning, quantization-aware optimization, and evaluation. It is designed to help organize and run repeated model-tuning workflows with configurable toggles, architecture awareness, rollback markers, and provenance tracking.

The project supports a staged tuning pipeline for models that need systematic evaluation and optimization across capability, efficiency, and safety dimensions.

## What it does

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

The package contains 99 tuning-toggle records. Fifteen have callable pipeline entrypoints; 84 named research modules remain unavailable prototypes. The CLI lists each status and rejects requests for unavailable stages before loading a model. Operational stages include live architecture inspection, refusal-direction ablation, optional Quanto quantization, numeric parameter scanning, read-only MoE routing diagnostics, Pareto ranking of caller-supplied candidate scores, pre-edit causal-LM evaluation baselines, a direction-aware post-edit quality/forward-throughput regression gate, prompt-redacted run-audit artifacts, optional fingerprint/provenance/reproducibility/memory/model-card reports, and separate validated Transformers export. The routing stage does not rebalance experts; the numeric scan is not malware/backdoor detection; and Pareto selection does not generate or evaluate candidates. The new evaluation gate is a limited language-modeling/forward-throughput check, not a hallucination, reasoning, safety, or deployment guarantee. Runtime and memory reports are observational; reproducibility reporting does not set random seeds or promise bitwise determinism. Do not interpret a prototype's name or class file as evidence its feature is operational.

For status against the supplied implementation phases and a conservative release-readiness assessment, see [`docs/implementation-status.md`](docs/implementation-status.md).
The opt-in diagnostic, evaluation-gate, and audit-stage contracts are documented in [`docs/advanced-stages.md`](docs/advanced-stages.md). The implementation record for the latest three promoted stages and the full 84-entry remaining-prototype inventory are in [`docs/next-prototype-batch-2026-11-16.md`](docs/next-prototype-batch-2026-11-16.md) and [`docs/remaining-prototype-inventory-2026-11-16.csv`](docs/remaining-prototype-inventory-2026-11-16.csv).

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

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/NRAEnterprises/AUTOBLITERATUS.git
cd AUTOBLITERATUS
python -m pip install -e .
```

## Quick Start

List available tuning toggles:

```bash
autobliteratus --list-toggles
```

Run the pipeline on a model. By default, the package estimates refusal
directions, applies an attention-output projection to the in-memory model, and
writes a separately validated Transformers model under
`./abliterate_workspace/modified_model`:

```bash
autobliteratus --model-path /path/to/model --work-dir ./abliterate_workspace
```

This ablation intentionally changes refusal-related behavior and may weaken
safety safeguards. Use only models you are authorized to modify, isolate the
result, and independently evaluate safety and target-task quality before any
use; successful export is not approval for deployment.

Choose a different output location or projection target:

```bash
autobliteratus --model-path /path/to/model \
  --output-dir /path/to/new-model-copy \
  --ablation-component mlp.down --ablation-strength 0.8
```

The output directory must not exist. The exporter writes into a temporary
sibling directory, verifies that the saved configuration and tokenizer load,
and checks Safetensors tensor names, shapes, values, and required-key coverage
against the in-memory model before publishing the directory with a rename. The
source model directory is never used as an output location; overlapping paths
are rejected. Existing output directories are never overwritten.

Run calibrated activation quantization using representative text (plain text,
JSONL rows with a `text` or `prompt` field, or a JSON list):

```bash
autobliteratus --model-path /path/to/model \
  --quantization quanto-int4 \
  --calibration-data /path/to/representative-text.jsonl \
  --calibration-samples 128 \
  --calibration-max-length 256 \
  --output-dir /path/to/new-quantized-model
```

`quanto-int4` and `quanto-int8` quantize backend-supported linear weights and
calibrate int8 activation ranges from the supplied corpus. The corpus itself is
not copied to the output; its SHA-256 and calibration settings are recorded in
the export manifest. The language-model output head is excluded by default.
Each text row currently forms one Quanto calibration update, so the moving-
average momentum and row order can influence measured activation scales. Test
representative data with deployment-like inputs and review saved calibration
settings rather than treating defaults as universal.
`quanto-int4-weight-only` and `quanto-int8-weight-only` omit activation
quantization and require no calibration corpus. These modes help separate
weight-quantization effects from activation-calibration effects, but are not
supported for every model module type. Before reporting success, the pipeline
checks the backend map contains at least one quantized weight and that its bit
type matches the requested mode; a model with only activation-map entries is
rejected. In the tested Quanto release, GPT-2 `Conv1D` weights are not covered
by the supported weight conversion, so its requested weight quantization fails
closed rather than silently producing an activation-only artifact.

Reload exported Quanto models with the package compatibility loader:

```python
from autobliteratus.quantization_compat import load_quantized_model

model = load_quantized_model("/path/to/new-quantized-model", device="cpu")
```

This checks the checkpoint against the instantiated architecture and reconciles
legacy GPT-NeoX output-head tensor names that Quanto 0.2.7 can otherwise ignore
silently when loading with newer Transformers releases. The benchmark runner
and integration tests use this same loader. For architectures that tie the
output head to input embeddings, the loader also accepts an omitted duplicate
head tensor only when the config declares tying, the saved embedding tensor is
present with an exact shape match, and the output head is unquantized; other
missing or unexpected keys still fail closed.

Run a paired hardware-local benchmark over four model families and six sizes
by default (Pythia 70M, SmolLM2 135M/360M, GPT-2 124M, and Qwen2.5 0.5B/1.5B):

```bash
python benchmarks/quantization_quality_matrix.py \
  --output-dir ./benchmark-results/arm64-run \
  --seed 2026 \
  --device auto
```

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
calibration data. See [`benchmarks/hardware-matrix.md`](benchmarks/hardware-matrix.md)
for the device/model matrix and per-platform caveats.

Enable or disable specific tuning modules:

```bash
autobliteratus --model-path /path/to/model \
  --enable tuning.weight_fingerprint_auto tuning.provenance_recording_auto \
  --disable tuning.refusal_direction_ablation
```

Stages with only a class API and no pipeline-compatible module-level
`run(adapter, context)` function are prototypes and are rejected when explicitly
enabled or disabled; they are never reported as executed. The architecture
inspector reports observed structure and does not assert backend support. The
exporter records the deployment-profile metadata but does not claim profile-
specific optimization has occurred.

At this repository revision, 12 of 99 registered tuning toggles have callable
pipeline stages; the other 87 are deliberately unavailable and are not
production implementations. This is a remaining release blocker for any
workflow that depends on those capabilities; their names are not evidence that
the behavior has been implemented.

Dry-run a configuration without executing full tuning:

```bash
autobliteratus --model-path /path/to/model --dry-run
```

You can also invoke the package directly:

```bash
python -m autobliteratus --model-path /path/to/model --work-dir ./abliterate_workspace
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

See [`docs/run-plans-and-sharding.md`](docs/run-plans-and-sharding.md) and the
[`example YAML run plan`](configs/run-plan.example.yaml) for precedence, safety,
multi-device placement, and sharded export details. Placement/sharding controls
are implemented but are not a claim of tested acceleration on every backend;
the current development host is ARM64 CPU-only.

## Repository layout

- `src/autobliteratus/` - package source code
- `pyproject.toml` - project metadata and Python package configuration
- `archive/root-prototypes-2026-11-16/` - preserved root-level experimental modules; not imported or packaged as active features
- `Makefile` - repeatable install, test, lint, benchmark-help, and build targets
- `testing/TESTING_LOG.md` and `testing/results.csv` - append-only test and benchmark ledger
- `docs/run-plans-and-sharding.md` - config, plan-only, Accelerate, offload, and sharded export guide
- `README.md` - project overview and usage
- `LICENSE` - MIT license
- `abliterate_workspace/` - working directory for generated artifacts and reports
- `provenance/` and `reproducibility/` - output and tracking directories
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
Append every new test or benchmark result—including failed, invalid, and
not-run cases—to [`testing/results.csv`](testing/results.csv), following the
rules in [`testing/TESTING_LOG.md`](testing/TESTING_LOG.md).

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
