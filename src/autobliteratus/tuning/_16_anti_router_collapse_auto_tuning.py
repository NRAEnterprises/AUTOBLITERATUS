"""Tuning type 16: anti router collapse auto tuning.

For MoE models, measure routing entropy and restore expert utilization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


TOGGLE = {"id": "tuning.anti_router_collapse", "default": True}


@dataclass
class RouterCollapseReport:
    entropy_per_layer: dict = field(default_factory=dict)
    collapsed_layers: list = field(default_factory=list)
    rebalanced_layers: list = field(default_factory=list)


class AntiRouterCollapseAutoTuning:
    ENTROPY_FLOOR = 1.5

    def __init__(self, rebalance: Callable, enabled: bool = True) -> None:
        self.rebalance = rebalance
        self.enabled = enabled

    def run(self, model) -> RouterCollapseReport:
        if not self.enabled:
            return RouterCollapseReport()
        entropy = self._measure_entropy(model)
        collapsed = [i for i, e in entropy.items() if e < self.ENTROPY_FLOOR]
        rebalanced = []
        for layer_idx in collapsed:
            self.rebalance(model, layer_idx)
            rebalanced.append(layer_idx)
        return RouterCollapseReport(
            entropy_per_layer=entropy,
            collapsed_layers=collapsed,
            rebalanced_layers=rebalanced,
        )

    def _measure_entropy(self, model) -> dict:
        return {}
