# Phase A: Contract and Shared Infrastructure Specification

## Purpose

Phase A is the mandatory promotion gate for every unavailable prototype switch. It does not enable new model behavior. It establishes the shared contracts that prevent a prototype from becoming a new breakage point.

## 1. Stage contract

Every promoted stage must expose a module-level callable:

```python
def run(adapter: Any, context: dict[str, Any]) -> dict[str, Any]:
    ...
```

Required behavior:

- Accept the repository's adapter shape and use `adapter.handle.model` only through documented adapter access.
- Read declared inputs from `context`; do not reach into global state or infer undocumented options.
- Return a JSON-serializable report with a stable `stage_id`, `status`, and `schema_version`.
- Write all derived values back into a namespaced context key such as `context["<stage_name>"]`.
- Declare whether the stage is diagnostic-only or may mutate model weights.
- Fail closed on missing, non-finite, incompatible, or ambiguous inputs.
- Never silently substitute a prototype class method for the required module-level `run` entrypoint.

Minimum report schema:

```json
{
  "stage_id": "tuning.example_stage",
  "schema_version": 1,
  "status": "passed|failed|skipped",
  "mutates_model": false,
  "measurement_point": "before_edit|after_edit|before_export",
  "metrics": {},
  "artifacts": [],
  "warnings": []
}
```

## 2. Registry contract

The registry must remain statically discoverable without importing optional prototype dependencies.

- A module is callable only when its top-level `run` function is present.
- `TOGGLE["id"]` must be unique, stable, and equal to the CLI switch.
- `available` is derived from the top-level run entrypoint, not from the presence of classes.
- Unavailable switches must remain visible in `--list-toggles` and be labeled `[prototype: unavailable]`.
- Enabling an unavailable switch must raise a deterministic `ValueError` before model loading.
- No dependency edge may reference an unavailable prototype in the active execution graph.
- Historical baseline only: the original Phase A registry audit asserted 99 total records, 15 callable stages, and 84 unavailable prototypes. The current registry count is maintained in `PROJECT_HANDOFF_MANIFEST.yaml` and must be measured rather than copied from this dated contract.

## 3. Dependency and orchestration contract

- Dependencies are directed stage IDs, not module names or class names.
- Graph construction must reject cycles with the cycle's stage ID.
- A stage may execute only after resolved, callable dependencies have completed successfully.
- A failed dependency skips dependent work and records the reason.
- Independent reporting stages may continue after an unrelated stage fails.
- The orchestrator report must separate `executed`, `skipped`, and `failed` records.
- Stage failures must include `error_type`; raw exception text must not be written to redacted audit artifacts.

## 4. Context and artifact contract

- Context keys are namespaced and documented per stage.
- Artifact paths are explicit, unique, and located under the run output directory.
- A stage must not overwrite the source model or user-owned files.
- Writes use temporary siblings and atomic publication where the stage creates a durable artifact.
- Reports must include checksums for model-derived artifacts and version metadata for reproducibility.
- Raw prompts, evaluation text, secrets, and arbitrary CLI options must not be copied into audit artifacts.

## 5. Safety and compatibility contract

- Validate model family, tensor dtype, device placement, and required module layout before mutation.
- Reject unsupported architecture families instead of broad family-name guessing.
- Require finite metrics and bounded resource requests.
- Optional backends must fail with an actionable unavailable-backend reason, not a generic model defect.
- Diagnostic stages must declare `mutates_model=false` and have a negative test proving no parameter changed.
- Mutating stages must have before/after fingerprints or equivalent change evidence.

## 6. Required tests before promotion

Each switch must add:

1. Registry discovery test.
2. Positive unit test with a deterministic tiny fixture.
3. Missing-input or unsupported-architecture negative test.
4. Non-finite metric or resource-limit negative test where applicable.
5. Pipeline integration test proving ordering and context handoff.
6. Source-preservation test for any export or mutation path.
7. Artifact schema and checksum test for any file-producing stage.
8. Failure-isolation test proving independent stages remain observable.
9. Coverage threshold: at least 85% on the new stage, with explicit branch exceptions documented.

## 7. Promotion checklist

A switch may move from unavailable to callable only when the implementation, tests, documentation, coverage, registry counts, dependency graph, and release audit are updated in one change. Do not promote a stage because its prototype classes import successfully; promotion requires a proven `run(adapter, context)` contract and measurable acceptance evidence.
