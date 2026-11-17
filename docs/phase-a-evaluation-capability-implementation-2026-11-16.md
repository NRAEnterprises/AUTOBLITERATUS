# Phase A Implementation Record — Evaluation and Capability Quality

Date: 2026-11-16

## Scope

Phase A has started for the evaluation and capability quality workstream. The workstream contains 20 currently unavailable switches after the first Phase A promotion. The first implementation is the shared capability-battery contract, promoted through a deterministic adapter boundary rather than through the legacy class-only stub.

## Promoted Phase A backend

`Tuning type 6: tuning.auto_ablated_capability` now exposes `run(adapter, context)` and is opt-in by default. It accepts an explicit non-empty axis list, rejects unknown or duplicate axes, calls the adapter capability contract, validates finite scores in `[0, 1]`, writes only to `context["capability_battery_report"]`, and declares `model_modified=False`.

The stage is ordered after refusal-direction ablation when that stage is enabled. It does not claim to measure every capability axis equally: the underlying capability helper currently has substantive probes for coherence, reasoning, code, math, and refusal rate, while several other axes use placeholder scores. Those axes remain visible in the report but should not be marketed as validated evidence until dedicated probes are implemented.

## Phase A contract

```python
def run(adapter: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Return a finite, JSON-serializable capability report without mutation."""
```

Required context:

- `capability_axes`: optional list drawn from the declared axis vocabulary.
- adapter capability method: `adapter.capability(tuple(axes))`.

Required report:

```json
{
  "stage_id": "tuning.auto_ablated_capability",
  "schema_version": 1,
  "status": "measured",
  "scores": {"coherence": 0.0},
  "axes": ["coherence"],
  "sample_count": 1,
  "model_modified": false,
  "artifacts": [],
  "warnings": []
}
```

## Evidence

- Phase A capability tests: 4 passed.
- Full project suite after the change: run through the release validation target before promotion packaging.
- Static registry audit: the stage is callable only because the module-level `run` exists.
- Negative coverage: unknown axis, duplicate axis, non-finite score, and default-axis behavior are tested.

## Next evaluation-quality batch

1. Add dedicated, versioned probes for structured output, tool-call integrity, multilingual behavior, long-context retrieval, and multi-turn coherence.
2. Replace placeholder axis values in `core/capability.py` with measurable evaluators or mark those axes `not_evaluated` rather than returning synthetic 1.0 values.
3. Add baseline/post comparison contracts with uncertainty and metric-direction metadata for `post_ablation_capability`.
4. Add calibration and hallucination probes only with versioned datasets and explicit claim limitations.
5. Add code and math execution sandboxes with timeouts, resource limits, no network, and artifact cleanup.
6. Promote the next stage only after the universal security, integration, coverage, and wheel/sdist gates pass.

## Blocking condition

A quality stage must not be promoted merely because it returns a complete dictionary. Placeholder scores, unbounded evaluator execution, unversioned datasets, and unsupported architecture assumptions are Phase A blockers.
