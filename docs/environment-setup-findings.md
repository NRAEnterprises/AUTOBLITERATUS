# Development-tool setup findings — 2026-11-16

The inherited `.venv-auto` did not initially contain the declared development tools:

- First `make build`: failed with `No module named build`.
- First `make lint`: failed with `No module named ruff`.

These were environment setup failures, not source/build failures. The tools were installed into the virtual environment (`build>=1.2`, Ruff), after which:

- `make build` successfully created the wheel and source distribution (see `package-build-2026-11-16.log`).
- The focused `make lint` target passed (see `focused-lint-2026-11-16.log`).
- `make lint-all` remains failed because the latest active-repository scan has 184 Ruff findings, 170 auto-fixable (an earlier scan had 187); see `project-wide-lint-2026-11-16.log` and `results.csv`.
