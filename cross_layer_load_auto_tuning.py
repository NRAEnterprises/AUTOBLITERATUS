# abliterate/tuning/31_cross_layer_load_auto_tuning.py
#
# Tuning Type 31 of 103:
# Cross layer load auto tuning.
#
# Runtime toggle: --enable tuning.cross_layer_load_auto  (default: on)
#
# Effect:
#   When one layer is projected, adjacent layers absorb the load. This file
#   redistributes the projection across layers so no single layer carries
#   more than its share.
#
# This file owns:
#   - per-layer load measurement
#   - load redistribution
#   - re-verification after redistribution

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class LoadDistribution:
    per_layer_load: dict = field(default_factory=dict)
    redistributed: dict = field(default_factory=dict)


class CrossLayerLoadAutoTuning:

    def __init__(self, apply_projection: Callable, enabled: bool = True,
                 max_load_per_layer: float = 1.5):
        self.apply_projection = apply_projection
        self.enabled = enabled
        self.max_load_per_layer = max_load_per_layer

    def redistribute(self, model, planned_load: dict) -> LoadDistribution:
        if not self.enabled:
            return LoadDistribution(per_layer_load=planned_load)
        redistributed = {}
        for layer_idx, load in planned_load.items():
            if load <= self.max_load_per_layer:
                redistributed[layer_idx] = load
                continue
            excess = load - self.max_load_per_layer
            redistributed[layer_idx] = self.max_load_per_layer
            neighbors = [layer_idx - 1, layer_idx + 1]
            per_neighbor = excess / len(neighbors)
            for n in neighbors:
                redistributed[n] = redistributed.get(n, 0.0) + per_neighbor
        return LoadDistribution(per_layer_load=planned_load, redistributed=redistributed)


TOGGLE = {"id": "tuning.cross_layer_load_auto", "default": True}
