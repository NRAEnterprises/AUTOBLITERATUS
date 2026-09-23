"""Tuning type 54: cross-layer compensation auto tuning.

When one layer loses capability, adjacent layers absorb the responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


TOGGLE = {"id": "tuning.cross_layer_compensation", "default": True}


@dataclass
class CompensationPlan:
    reallocations: dict = field(default_factory=dict)


class CrossLayerCompensationAutoTuning:
    def __init__(self, apply_reallocation: Callable, enabled: bool = True) -> None:
        self.apply_reallocation = apply_reallocation
        self.enabled = enabled

    def run(self, model, per_layer_capability: dict) -> CompensationPlan:
        if not self.enabled:
            return CompensationPlan()
        reallocations = {}
        for idx, cap in per_layer_capability.items():
            if cap < 0.95:
                reallocations[idx] = {idx - 1: 0.5, idx + 1: 0.5}
        for idx, mapping in reallocations.items():
            self.apply_reallocation(model, idx, mapping)
        return CompensationPlan(reallocations=reallocations)
