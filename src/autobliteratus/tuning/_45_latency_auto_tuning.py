"""Tuning type 45: latency auto tuning.

First-token latency and per-token latency measured and optimized.
"""

from __future__ import annotations

from dataclasses import dataclass


TOGGLE = {"id": "tuning.latency_auto", "default": True}


@dataclass
class LatencyPlan:
    first_token_ms: float
    per_token_ms: float
    optimized: bool


class LatencyAutoTuning:
    def __init__(self, measure, optimize, enabled: bool = True) -> None:
        self.measure = measure
        self.optimize = optimize
        self.enabled = enabled

    def run(self, model, tokenizer) -> LatencyPlan:
        if not self.enabled:
            return LatencyPlan(0.0, 0.0, False)
        self.measure(model, tokenizer)
        self.optimize(model)
        after = self.measure(model, tokenizer)
        return LatencyPlan(
            first_token_ms=after.get("first_token_ms", 0.0),
            per_token_ms=after.get("per_token_ms", 0.0),
            optimized=True,
        )
