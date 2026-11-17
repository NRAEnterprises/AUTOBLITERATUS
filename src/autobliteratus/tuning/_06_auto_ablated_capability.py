"""Tuning type 6: auto abalated capability.

Define and run the capability battery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

TOGGLE = {
    "id": "tuning.auto_ablated_capability",
    "default": False,
    "desc": "Measure an explicit capability-axis battery after transformations without mutating model weights.",
}


AXES = (
    "coherence",
    "reasoning",
    "cot_integrity",
    "structure",
    "calibration",
    "workflow",
    "tokenization_consistency",
    "multiturn_state",
    "tool_call_integrity",
    "structured_output",
    "multilingual",
    "code",
    "math",
    "long_context",
    "refusal_rate",
    "jailbreak_resistance",
    "hallucination_rate",
    "bias_profile",
)


@dataclass
class CapabilityBatteryReport:
    scores: dict[str, float] = field(default_factory=dict)


class AutoAblatedCapability:
    def __init__(self, model=None, tokenizer=None) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def measure(self) -> CapabilityBatteryReport:
        return CapabilityBatteryReport(scores={axis: 0.0 for axis in AXES})

    def __call__(self, model, tokenizer) -> dict:
        self.model = model
        self.tokenizer = tokenizer
        return self.measure().scores


def run(adapter: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Measure the requested capability axes through the adapter contract."""
    axes = context.get("capability_axes", AXES)
    if not isinstance(axes, (list, tuple)) or not axes or not all(isinstance(axis, str) for axis in axes):
        raise TypeError("capability_axes must be a non-empty list of strings.")
    unknown = sorted(set(axes) - set(AXES))
    if unknown:
        raise ValueError(f"Unsupported capability axes: {unknown}")
    if len(set(axes)) != len(axes):
        raise ValueError("capability_axes cannot contain duplicates.")
    scores = adapter.capability(tuple(axes))
    if not isinstance(scores, dict) or set(scores) != set(axes):
        raise ValueError("Capability evaluator must return exactly one score per requested axis.")
    validated: dict[str, float] = {}
    for axis, score in scores.items():
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError(f"Capability score for {axis!r} must be finite.")
        if not 0.0 <= float(score) <= 1.0:
            raise ValueError(f"Capability score for {axis!r} must be in [0, 1].")
        validated[axis] = float(score)
    report = {
        "stage_id": "tuning.auto_ablated_capability",
        "schema_version": 1,
        "status": "measured",
        "scores": validated,
        "axes": list(axes),
        "sample_count": len(axes),
        "model_modified": False,
        "artifacts": [],
        "warnings": [],
    }
    context["capability_battery_report"] = report
    return report
