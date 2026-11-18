# Implementation status

This page is the canonical implementation-status summary. It distinguishes code that exists from capabilities that have been tested or qualified.

## Current structure

The installable package is `src/autobliteratus/`, as declared in `pyproject.toml`. Runtime toggle discovery scans Python modules directly under `src/autobliteratus/tuning/`, reads literal module-level `TOGGLE` dictionaries, and marks a toggle callable only when that module has a top-level `run` function. Root-level Python files are outside that discovery and packaging boundary.

For current counts, run `python scripts/audit_repository_state.py`. The command's static inventory is authoritative for toggle totals only; it does not certify behaviors.

## Evidence boundaries

- **Implemented code:** inspect the package source and its tests.
- **Callable registry stage:** module has the entry point accepted by static discovery.
- **Tested behavior:** must be supported by a test result tied to a specific commit and environment.
- **Backend/hardware qualification:** must have target-specific run evidence; generic placement support is not qualification.
- **Production readiness:** not established merely by tests passing, package building, or a feature appearing in the registry.

Dated status reports elsewhere in the repository contain historical snapshots with different test counts, coverage percentages, lint results, and toggle totals. Preserve those reports as history, but cite their date and commit. Do not merge their numbers into a current claim without rerunning the referenced checks.

## Current conservative posture

The project is experimental. The package provides a model-loading and transformation pipeline, evaluation/diagnostic stages, and validated export paths, but the available documentation itself identifies gaps in coverage, accelerator qualification, optional quantization backend qualification, and failure/reporting consistency. Do not claim general production readiness or universal model/backend support.

## Reproduction

From the repository root:

```bash
python scripts/audit_repository_state.py
make test
make lint
make lint-all
make build
```

Record output, exact commit, platform, dependency versions, and skipped tests in `testing/`. A command not run is not a pass; an environment-skipped test is not backend qualification.
