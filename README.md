# AUTOBLITERATUS

AUTOBLITERATUS is an experimental Python toolkit for model ablation, evaluation, and quantization workflows. The installable application is the `autobliteratus` package under `src/autobliteratus/`; root-level Python scripts are preserved legacy/prototype files and are not the active packaged runtime.

## Source of truth

- **Current repository layout and status rules:** [docs/REPOSITORY_STATE.md](docs/REPOSITORY_STATE.md)
- **Implementation status:** [docs/implementation-status.md](docs/implementation-status.md)
- **Pipeline and optional stages:** [docs/advanced-stages.md](docs/advanced-stages.md)
- **Run plans, placement, and sharded exports:** [docs/run-plans-and-sharding.md](docs/run-plans-and-sharding.md)
- **Hardware evaluation matrix:** [benchmarks/hardware-matrix.md](benchmarks/hardware-matrix.md)
- **Dated audit and benchmark evidence:** [docs/](docs/)

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

Run against a model you are authorized to modify, using a distinct output directory:

```bash
autobliteratus --model-path /path/to/model \
  --work-dir ./abliterate_workspace \
  --output-dir ./model-copy
```

Ablation intentionally changes model behavior and can weaken safety safeguards. Evaluate the resulting model independently; a successful export is not a deployment approval or a quality/safety guarantee. The source model should remain separate from the output.

Preview an example run plan without loading model weights:

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

The test ledger is in [testing/TESTING_LOG.md](testing/TESTING_LOG.md) and [testing/results.csv](testing/results.csv). Record the commit, environment, exact command, and outcomes; do not treat an old result as a current run.

## Repository map

- `src/autobliteratus/` — canonical package source.
- `tests/` — automated tests and fixtures.
- `benchmarks/` — benchmark tools and result artifacts.
- `configs/` — example and project configuration.
- `docs/` — user/developer documentation and dated evidence.
- `ui/model-mapping-observatory/` — separate UI application.
- `legacy_root/` — reserved destination for any future fully reviewed legacy relocation; root-level prototypes are not yet represented there and must not be assumed moved.

See [docs/REPOSITORY_STATE.md](docs/REPOSITORY_STATE.md) for audited paths, known legacy naming discrepancies, and the relocation policy.
