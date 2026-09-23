# abliterate/tuning/38_layernorm_rmsnorm_auto_tuning.py
#
# Tuning Type 38 of 103:
# LayerNorm / RMSNorm auto tuning.
#
# Runtime toggle: --enable tuning.layernorm_rmsnorm_auto  (default: on)
#
# Effect:
#   Normalization statistics can drift after weight surgery and cause silent
#   instability. Measure them against baseline and restore them.
#
# This file owns:
#   - the norm-statistics drift measurement
#   - the norm-statistics restoration

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class NormProfile:
    drift_per_layer: dict
    restored_layers: list


class LayerNormRMSNormAutoTuning:

    def __init__(self, measure: Callable, apply_correction: Callable,
                 enabled: bool = True):
        self.measure = measure
        self.apply_correction = apply_correction
        self.enabled = enabled

    def run(self, model, baseline_stats) -> NormProfile:
        if not self.enabled:
            return NormProfile({}, [])
        drift = self.measure(model, baseline_stats)
        restored = []
        for layer, d in drift.items():
            if abs(d) > 0.01:
                self.apply_correction(model, layer, d)
                restored.append(layer)
        return NormProfile(drift_per_layer=drift, restored_layers=restored)


TOGGLE = {"id": "tuning.layernorm_rmsnorm_auto", "default": True}
