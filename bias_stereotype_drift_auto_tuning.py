# abliterate/tuning/62_bias_stereotype_drift_auto_tuning.py
#
# Tuning Type 62 of 103:
# Bias / stereotype drift auto tuning.
#
# Runtime toggle: --enable tuning.bias_stereotype_drift_auto  (default: on)
#
# Effect:
#   Ablation can shift bias profiles. Measure and correct.
#
# This file owns:
#   - bias profile measurement
#   - bias profile correction

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BiasProfile:
    per_category_drift: dict = field(default_factory=dict)
    corrected: bool = False


class BiasStereotypeDriftAutoTuning:

    def __init__(self, measure: Callable, apply_correction: Callable,
                 enabled: bool = True):
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


TOGGLE = {"id": "tuning.bias_stereotype_drift_auto", "default": True}
