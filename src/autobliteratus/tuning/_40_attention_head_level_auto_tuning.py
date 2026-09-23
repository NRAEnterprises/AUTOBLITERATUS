"""Tuning type 40: attention head-level auto tuning.

Prune or reweight individual heads rather than whole layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


TOGGLE = {"id": "tuning.attention_head_level_auto", "default": True}


@dataclass
class HeadPlan:
    pruned: list = field(default_factory=list)
    reweighted: dict = field(default_factory=dict)


class AttentionHeadLevelAutoTuning:
    def __init__(self, measure: Callable, apply: Callable, enabled: bool = True) -> None:
        self.measure = measure
        self.apply = apply
        self.enabled = enabled

    def run(self, model) -> HeadPlan:
        if not self.enabled:
            return HeadPlan()
        importance = self.measure(model)
        pruned = [h for h, s in importance.items() if s < 0.01]
        for h in pruned:
            self.apply(model, h)
        return HeadPlan(pruned=pruned)
