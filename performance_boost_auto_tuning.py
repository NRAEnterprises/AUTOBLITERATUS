# abliterate/tuning/15_performance_boost_auto_tuning.py
#
# Tuning Type 15 of 103:
# Performance boost auto tuning.
#
# Runtime toggle: --enable tuning.performance_boost_auto  (default: on)
#
# Effect:
#   Measure inference speed and memory footprint after ablation. Apply
#   layer-level optimizations that preserve the ablation and the restored
#   capabilities.
#
# This file owns:
#   - speed measurement
#   - memory measurement
#   - selection of optimizations that do not undo the ablation or the
#     restored capabilities

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class PerfProfile:
    tokens_per_second: float = 0.0
    peak_memory_mb: float = 0.0
    first_token_latency_ms: float = 0.0


class PerformanceBoostAutoTuning:

    def __init__(self, measure: Callable, apply_optimization: Callable,
                 enabled: bool = True):
        self.measure = measure
        self.apply_optimization = apply_optimization
        self.enabled = enabled

    def run(self, model, tokenizer) -> PerfProfile:
        if not self.enabled:
            return PerfProfile()
        before = self.measure(model, tokenizer)
        self.apply_optimization(model)
        after = self.measure(model, tokenizer)
        return after


TOGGLE = {"id": "tuning.performance_boost_auto", "default": True}
