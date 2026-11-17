# Repository structure and state

## Authority and scope

This document records the repository layout and explains which sources determine current implementation status. It is not a claim that every feature described by a historical prototype is implemented.

The repository snapshot inspected for this audit is `main` at commit `eeed0f0a3875762fcbc4bcc7bc83388dfe38c948` (2026-11-17). Counts and test totals in older reports are historical snapshots, not automatically current facts. Re-run the audit script from the repository root before quoting current registry counts.

## Canonical runtime and source layout

- `src/autobliteratus/` — installable Python package, configured by `pyproject.toml` using the `src` layout.
- `src/autobliteratus/core/` — model adapters, loading, architecture discovery, and core pipeline support.
- `src/autobliteratus/tuning/` — toggle modules, pipeline stages, dispatch, orchestration, metrics, and mutation controls. Static discovery examines this directory; it does not discover root-level modules.
- `tests/` — pytest suite and fixtures.
- `benchmarks/` — benchmark programs and checked-in benchmark results.
- `configs/` — checked-in example/runtime configuration files.
- `docs/` — implementation notes, reports, plans, contracts, and evidence.
- `ui/model-mapping-observatory/` — separate frontend project.

The root-level Python files are not part of the installed package according to `pyproject.toml` package discovery. They are retained legacy/prototype material; their presence does not make them active pipeline stages. Do not move them into the package or delete them based only on names. First establish whether each has unique content, references, or historical value. A future relocation must preserve the complete original file and update references atomically.

## What the repository currently establishes

- The package uses Python 3.10+ and setuptools' `src` package layout.
- The CLI entry point is `autobliteratus = autobliteratus.__main__:main`.
- Tuning-stage discovery reads literal `TOGGLE` metadata from Python modules directly under `src/autobliteratus/tuning/` and marks a stage available when the module defines a top-level `run` function.
- Stage dispatch imports only stages marked available. A class-based prototype without a module-level entry point is not thereby a callable pipeline stage.
- The main workflow loads and modifies a model in memory, then exports to a separate destination; the README documents path validation and export checks.
- The benchmark/hardware documentation describes test targets, not proof that every listed accelerator/backend was qualified.
- Historical audit documents report differing test, coverage, registry, lint, and hardware snapshots. Those figures must retain their dates and must not be presented as one current result without rerunning the corresponding command.

## Known documentation/path discrepancies at the audited snapshot

The existing README and documents reference these paths that were absent from the audited tree:

- `docs/implementation-status.md`
- `docs/advanced-stages.md`
- `docs/run-plans-and-sharding.md`
- `configs/run-plan.example.yaml`
- `testing/TESTING_LOG.md`
- `testing/results.csv`
- `benchmarks/hardware-matrix.md`

This change set adds canonical versions or navigational documents for those paths rather than silently dropping the referenced topics. Historical files with spaces and inconsistent capitalization remain preserved; links should migrate to the canonical names over time.

The default work directory in package code is `./abliterate_workspace`. The committed directory is currently misspelled `abliterrate_workspace`; treat it as a legacy tracked artifact, not as the runtime default. Do not write new outputs into that tracked directory.

## Current status procedure

Run:

```bash
python scripts/audit_repository_state.py
```

The script derives toggle totals from the same directory and literal-metadata rule used by runtime discovery. It also reports root-level Python files and duplicate/missing toggle metadata. The result is a static inventory, not functional qualification: tests, backend availability, hardware qualification, and quality claims require their own commands and evidence.

For tests, run `make test`; for focused lint and full lint, use `make lint` and `make lint-all`; for package distributions, use `make build`. Record results with the exact command, date, environment, commit, and pass/skip/fail counts. Do not copy historical totals forward.

## Layout policy for future changes

1. Active importable runtime code belongs in `src/autobliteratus/`.
2. Tests belong in `tests/`; benchmark executables and their outputs belong in `benchmarks/` and `benchmarks/results/` respectively.
3. User-facing how-to documentation belongs in `docs/`; canonical examples belong under `configs/`.
4. Runtime outputs belong in a caller-selected work/output directory and should not be committed as source artifacts unless explicitly curated as evidence.
5. Root-level prototypes remain preserved until their content and dependencies have been reviewed. Any later archive move must be a complete move, not a partial copy, and must preserve provenance and update references.
6. `README.md` is the project entry point. Detailed claims should link to the relevant current implementation, test, or dated evidence. A historical report is not current status.
