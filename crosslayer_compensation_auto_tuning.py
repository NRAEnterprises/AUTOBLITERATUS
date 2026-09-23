# abliterate/tuning/54_cross_layer_compensation_auto_tuning.py
#
# Tuning Type 54 of 103:
# Cross-layer compensation auto tuning.
#
# Runtime toggle: --enable tuning.cross_layer_compensation  (default: on)
#
# Effect:
#   Same target as tuning/31 but compensating for capability loss rather than
#   projection load. When one layer loses capability, adjacent layers absorb
#   the responsibility.
#
# This file owns:
#   - capability loss detection per layer
#   - responsibility reallocation across adjacent layers

from __future__ import annotationsfrom dataclasses import dataclass, field
from typing import Callable


@dataclass
class CompensationPlan:
    reallocations: dict = field(default_factory=dict)


class CrossLayerCompensationAutoTuning:

    def __init__(self, apply_reallocation: Callable, enabled: bool = True):
        self.apply_reallocation = apply_reallocation
        self.enabled = enabled

    def run(self, model, per_layer_capability: dict) -> CompensationPlan:
        if not self.enabled:
            return CompensationPlan()
        reallocations = {}
        for idx, cap in per_layer_capability.items():
            if cap < 0.95:
                reallocations[idx] = {idx - 1: 0.5, idx + 1: 0.5}
        for idx, map_ in reallocations.items():
            self.apply_reallocation(model, idx, map_)
        return CompensationPlan(reallocations=reallocations)


TOGGLE = {"id": "tuning.cross_layer_compensation", "default": True}
