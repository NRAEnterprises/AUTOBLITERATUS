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

This project is experimental and under active development. Some modules and workflow integrations may be incomplete or change as the toolkit evolves.

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

Run the pipeline on a model:

```bash
autobliteratus --model-path /path/to/model --work-dir ./abliterate_workspace
```

Enable or disable specific tuning modules:

```bash
autobliteratus --model-path /path/to/model --enable quantization_aware_capability --disable latency_auto_tuning
```

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
- `--work-dir` - scratch/output directory for generated artifacts
- `--quantization` - quantization mode (`none`, `4bit`, `8bit`, `fp8`, `nvfp4`, `mxfp4`)
- `--target-frameworks` - target compatibility frameworks
- `--deployment-target` - deployment profile (`server`, `desktop`, `edge`, `mobile`)
- `--enable` - toggle IDs to force on
- `--disable` - toggle IDs to force off
- `--list-toggles` - list registered tuning modules
- `--dry-run` - validate configuration without running the full pipeline

## Repository layout

- `src/autobliteratus/` - package source code
- `pyproject.toml` - project metadata and Python package configuration
- `pipeline_entry.py` - pipeline entrypoint prototype for orchestration logic
- `README.md` - project overview and usage
- `LICENSE` - MIT license
- `abliterate_workspace/` - working directory for generated artifacts and reports
- `provenance/` and `reproducibility/` - output and tracking directories
- `tests/` - test coverage area

## Example workflow

1. Select a target model.
2. Choose a quantization and deployment profile.
3. Enable or disable specific tuning toggles.
4. Run `autobliteratus`.
5. Review generated artifacts, provenance, and reproducibility markers.

## Important note

This toolkit performs model transformation, optimization, and evaluation workflows. It should only be used on models and datasets that you are authorized to process, and it should be run in a controlled environment with proper validation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
