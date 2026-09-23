"""Tuning type 62: bias and stereotype drift auto tuning.

Ablation can shift bias profiles. Measure and correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


TOGGLE = {"id": "tuning.bias_stereotype_drift_auto", "default": True}


@dataclass
class BiasProfile:
    per_category_drift: dict = field(default_factory=dict)
    corrected: bool = False


class BiasStereotypeDriftAutoTuning:
    def __init__(self, measure: Callable, apply_correction: Callable, enabled: bool = True) -> None:
        self.measure = measure
        self.apply_correction = apply_correction
        self.enabled = enabled

    def run(self, model, tokenizer, baseline: dict) -> BiasProfile:
        if not self.enabled:
            return BiasProfile()
        current = self.measure(model, tokenizer)
        drift = {k: current.get(k, 0.0) - baseline.get(k, 0.0) for k in baseline}
        if any(abs(v) > 0.05 for v in drift.values()):
            self.apply_correction(model, drift)
            return BiasProfile(per_category_drift=drift, corrected=True)
        return BiasProfile(per_category_drift=drift, corrected=False)
