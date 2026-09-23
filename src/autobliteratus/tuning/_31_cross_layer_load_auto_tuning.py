"""Tuning type 31: cross layer load auto tuning.

Redistribute projection load across adjacent layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


TOGGLE = {"id": "tuning.cross_layer_load_auto", "default": True}


@dataclass
class LoadDistribution:
    per_layer_load: dict = field(default_factory=dict)
    redistributed: dict = field(default_factory=dict)


class CrossLayerLoadAutoTuning:
    def __init__(
        self,
        apply_projection: Callable,
        enabled: bool = True,
        max_load_per_layer: float = 1.5,
    ) -> None:
        self.apply_projection = apply_projection
        self.enabled = enabled
        self.max_load_per_layer = max_load_per_layer

    def redistribute(self, model, planned_load: dict) -> LoadDistribution:
        if not self.enabled:
            return LoadDistribution(per_layer_load=planned_load)
        redistributed: dict = {}
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
