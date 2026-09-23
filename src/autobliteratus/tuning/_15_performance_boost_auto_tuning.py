"""Tuning type 15: performance boost auto tuning.

Measure inference speed and memory footprint after ablation. Apply
optimizations that preserve the ablation and the restored capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


TOGGLE = {"id": "tuning.performance_boost_auto", "default": True}


@dataclass
class PerfProfile:
    tokens_per_second: float = 0.0
    peak_memory_mb: float = 0.0
    first_token_latency_ms: float = 0.0


class PerformanceBoostAutoTuning:
    def __init__(self, measure: Callable, apply_optimization: Callable, enabled: bool = True) -> None:
        self.measure = measure
        self.apply_optimization = apply_optimization
        self.enabled = enabled

    def run(self, model, tokenizer) -> PerfProfile:
        if not self.enabled:
            return PerfProfile()
        self.measure(model, tokenizer)
        self.apply_optimization(model)
        return self.measure(model, tokenizer)
