"""Tuning type 38: LayerNorm / RMSNorm auto tuning.

Normalization statistics can drift after weight surgery and cause silent
instability. Measure against baseline and restore.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


TOGGLE = {"id": "tuning.layernorm_rmsnorm_auto", "default": True}


@dataclass
class NormProfile:
    drift_per_layer: dict
    restored_layers: list


class LayerNormRMSNormAutoTuning:
    def __init__(self, measure: Callable, apply_correction: Callable, enabled: bool = True) -> None:
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
