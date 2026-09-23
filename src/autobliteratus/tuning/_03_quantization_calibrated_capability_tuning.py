"""Tuning type 3: quantization calibrated capability tuning.

Same loop as type 1, but every measurement and every correction runs at the
target quantization scheme for each layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


TOGGLE = {"id": "tuning.quantization_calibrated_capability", "default": True}


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
    scheme_map: dict[int, str]
    iterations: int
    restored: dict[str, float] = field(default_factory=dict)
    unrecovered: dict[str, dict] = field(default_factory=dict)
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
    ) -> None:
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

        restored: dict[str, float] = {}
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
            enabled=True,
            scheme_map=scheme_map,
            iterations=iteration + 1,
            restored=restored,
            unrecovered=unrecovered,
            ceiling_hit=bool(unrecovered),
        )

    def _select_layer(self, model, target) -> int:
        return self._num_layers(model) // 2

    def _build_targets(self, baseline: dict, current: dict) -> list[TuningTarget]:
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

    def _num_layers(self, model) -> int:
        if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
            return model.config.num_hidden_layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return len(model.model.layers)
        return 0
