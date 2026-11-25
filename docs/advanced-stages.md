# Advanced and optional stages

This is the canonical index for optional pipeline stages. A name in the toggle inventory is not evidence that the feature is a complete implementation. The runtime registry is derived from static `TOGGLE` metadata and a module-level `run` entry point under `src/autobliteratus/tuning/`.

## Inspect actual availability

Run:

```bash
python scripts/audit_repository_state.py
```

The generated totals report callable entry points and unavailable prototypes separately. It does not evaluate whether an available stage's behavior meets a broader production contract.

## Stage behavior and limits

- Read stage code and tests under `src/autobliteratus/tuning/` and `tests/` before relying on an option.
- Optional evaluation, diagnostics, selection, and profiling may require explicitly supplied data or callbacks; inspect each stage's input contract.
- A diagnostic or selector is not automatically an optimizer. A report-only stage should not be described as mutating or improving the model.
- Security-related numeric checks and refusal/probe measurements have narrow scopes. Do not generalize them to malware detection, safety certification, or broad adversarial robustness.
- Model and hardware compatibility claims require tests on the relevant architecture, backend, and versions.

The detailed historical contracts and audit evidence remain under `docs/`, including `Opt-in diagnostic and selection stages.md` and `phase-c-error-emission-audit-2026-11-17.md`. Their dates and documented limitations remain part of the evidence.
