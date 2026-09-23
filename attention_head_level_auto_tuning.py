# abliterate/tuning/40_attention_head_level_auto_tuning.py
#
# Tuning Type 40 of 103:
# Attention head-level auto tuning.
#
# Runtime toggle: --enable tuning.attention_head_level_auto  (default: on)
#
# Effect:
#   Prune or reweight individual heads rather than whole layers.
#
# This file owns:
#   - per-head importance measurement
#   - head pruning
#   - head reweighting

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class HeadPlan:
    pruned: list = field(default_factory=list)
    reweighted: dict = field(default_factory=dict)


class AttentionHeadLevelAutoTuning:

    def __init__(self, measure: Callable, apply: Callable, enabled: bool = True):
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


TOGGLE = {"id": "tuning.attention_head_level_auto", "default": True}
