# Project-wide lint remediation report

## Before

The project-wide command `ruff check src tests benchmarks` reported **177 diagnostics**, while the focused active-code gate passed.

## Cleanup executed

- Ran Ruff safe autofixes across `src`, `tests`, and `benchmarks`.
- Ran the explicitly available unsafe autofix only for the mechanical `startswith` tuple simplification.
- Corrected import ordering, deprecated typing aliases, unused imports, sorted `__all__`, nested conditionals, benchmark closure binding, and class-level mutable-constant annotations.
- Added explicit reviewed `BLE001` exceptions only where arbitrary framework/exporter/stage callbacks must be isolated to preserve independent results.
- Corrected the registry dispatch contract to raise `TypeError` for a callable-entrypoint violation.
- Made the benchmark CLI executable so its shebang is valid.
- Added backend tests for the two newly promoted compatibility stages.

## After

- Project-wide Ruff: **passes with no findings**.
- Focused active-code Ruff: **passes with no findings**.
- Full tests: **159 passed, 17 skipped**.
- Build: **wheel and source distribution pass**.

## Remaining policy considerations

The broad callback exception boundaries are intentional but should eventually be replaced with typed plugin/result contracts. The legacy prototype tree now lint-passes, but it remains unavailable runtime functionality until it receives the Phase A promotion contract, positive/negative tests, and measurable acceptance evidence.
