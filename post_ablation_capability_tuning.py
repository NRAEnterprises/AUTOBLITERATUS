# abliterate/tuning/01_post_ablation_capability_tuning.py
#
# Tuning Type 1 of 103:
# Post ablation capability tuning.
#
# Runtime toggle: --enable tuning.post_ablation_capability  (default: on)
#
# Effect:
#   Measure capability axes against the model's own pre-ablation baseline.
#   For every axis below baseline, apply a correction. Re-measure.
#   Exit when every axis is inside tolerance or the iteration ceiling is hit.
#
# This file owns nothing except the loop.
# It knows no architecture. It knows no quantization. It knows no corrections
# beyond calling the callable it was handed. It knows no rollback. It knows no
# markers. It knows no security. It knows no telemetry.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class TuningTarget:
    axis: str
    baseline: float
    current: float
    deficit: float
    tolerance: float


@dataclass
class TuningReport:
    enabled: bool
    iterations: int
    restored: dict = field(default_factory=dict)
    unrecovered: dict = field(default_factory=dict)
    ceiling_hit: bool = False


class PostAblationCapabilityTuning:

    def __init__(
        self,
        measure: Callable,
        apply_correction: Callable,
        enabled: bool = True,
        max_iterations: int = 20,
        tolerance_ratio: float = 0.02,
    ):
        self.measure = measure
        self.apply_correction = apply_correction
        self.enabled = enabled
        self.max_iterations = max_iterations
        self.tolerance_ratio = tolerance_ratio

    def run(self, model, tokenizer, pre_ablation_baseline: dict) -> TuningReport:
        if not self.enabled:
            return TuningReport(enabled=False, iterations=0)

        current = self.measure(model, tokenizer)
        targets = self._build_targets(pre_ablation_baseline, current)

        restored = {}
        iteration = 0
        for iteration in range(self.max_iterations):
            active = [t for t in targets if t.axis not in restored]
            if not active:
                break

            for target in active:
                self.apply_correction(
                    model=model,
                    axis=target.axis,
                    deficit=target.deficit,
                )

            current = self.measure(model, tokenizer)

            for target in active:
                new_value = current.get(target.axis, 0.0)
                target.current = new_value
                target.deficit = max(0.0, target.baseline - new_value)
                if target.deficit <= target.tolerance:
                    restored[target.axis] = new_value

        unrecovered = {
            t.axis: {"baseline": t.baseline, "final": t.current, "deficit": t.deficit}
            for t in targets if t.axis not in restored
        }

        return TuningReport(
            enabled=True,
            iterations=iteration + 1,
            restored=restored,
            unrecovered=unrecovered,
            ceiling_hit=bool(unrecovered),
        )

    def _build_targets(self, baseline: dict, current: dict) -> list:
        out = []
        for axis, base_value in baseline.items():
            current_value = current.get(axis, 0.0)
            out.append(TuningTarget(
                axis=axis,
                baseline=base_value,
                current=current_value,
                deficit=max(0.0, base_value - current_value),
                tolerance=base_value * self.tolerance_ratio,
            ))
        return out


TOGGLE = {"id": "tuning.post_ablation_capability", "default": True}
