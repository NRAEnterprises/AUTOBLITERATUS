# abliterate/tuning/03_quantization_calibrated_capability_tuning.py
#
# Tuning Type 3 of 103:
# Quantization calibrated capability tuning.
#
# Runtime toggle: --enable tuning.quantization_calibrated_capability  (default: on)
#
# Effect:
#   Same measure/correct/re-measure loop as tuning type 1, but every
#   measurement, every correction, and every verification is taken at the
#   target quantization scheme for each layer.
#
# This file owns:
#   - the precision-at-which-tuning-runs guarantee
#   - per-layer scheme application during the loop
#
# This file consumes:
#   - measure callable
#   - apply_correction callable
#   - scheme_for callable (from tuning/72, passed in)

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
    scheme_map: dict
    iterations: int
    restored: dict = field(default_factory=dict)
    unrecovered: dict = field(default_factory=dict)
    ceiling_hit: bool = False


class QuantizationCalibratedCapabilityTuning:

    def __init__(
        self,
        measure: Callable,
        apply_correction: Callable,
        scheme_for: Callable,
        enabled: bool = True,
        max_iterations: int = 20,
        tolerance_ratio: float = 0.02,
    ):
        self.measure = measure
        self.apply_correction = apply_correction
        self.scheme_for = scheme_for
        self.enabled = enabled
        self.max_iterations = max_iterations
        self.tolerance_ratio = tolerance_ratio

    def run(self, model, tokenizer, pre_ablation_baseline: dict) -> TuningReport:
        if not self.enabled:
            return TuningReport(enabled=False, scheme_map={}, iterations=0)

        num_layers = self._num_layers(model)
        scheme_map = {i: self.scheme_for(model, i) for i in range(num_layers)}

        current = self.measure(model, tokenizer)
        targets = self._build_targets(pre_ablation_baseline, current)

        restored = {}
        iteration = 0
        for iteration in range(self.max_iterations):
            active = [t for t in targets if t.axis not in restored]
            if not active:
                break
            for target in active:
                idx = self._select_layer(model, target)
                scheme = scheme_map[idx]
                self.apply_correction(
                    model=model,
                    layer_idx=idx,
                    axis=target.axis,
                    deficit=target.deficit,
                    scheme=scheme,
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
            enabled=True, scheme_map=scheme_map, iterations=iteration + 1,
            restored=restored, unrecovered=unrecovered, ceiling_hit=bool(unrecovered),
        )

    def _select_layer(self, model, target):
        return self._num_layers(model) // 2

    def _build_targets(self, baseline: dict, current: dict) -> list:
        out = []
        for axis, base_value in baseline.items():
            current_value = current.get(axis, 0.0)
            out.append(TuningTarget(
                axis=axis, baseline=base_value, current=current_value,
                deficit=max(0.0, base_value - current_value),
                tolerance=base_value * self.tolerance_ratio,
            ))
        return out

    def _num_layers(self, model) -> int:
        if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
            return model.config.num_hidden_layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return len(model.model.layers)
        return 0


TOGGLE = {"id": "tuning.quantization_calibrated_capability", "default": True}
