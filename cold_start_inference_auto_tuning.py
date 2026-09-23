# abliterate/tuning/98_cold_start_inference_auto_tuning.py
#
# Tuning Type 98 of 103:
# Cold-start inference auto tuning.
#
# Runtime toggle: --enable tuning.cold_start_inference_auto  (default: on)
#
# Effect:
#   Optimize the first-call latency after model load.
#
# This file owns:
#   - cold-start latency measurement
#   - cold-start optimization

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class ColdStartProfile:
    latency_ms: float
    optimized: bool


class ColdStartInferenceAutoTuning:

    def __init__(self, measure: Callable, optimize: Callable,
                 enabled: bool = True):
        self.measure = measure
        self.optimize = optimize
        self.enabled = enabled

    def run(self, model, tokenizer) -> ColdStartProfile:
        if not self.enabled:
            return ColdStartProfile(0.0, False)
        before = self.measure(model, tokenizer)
        self.optimize(model)
        after = self.measure(model, tokenizer)
        return ColdStartProfile(latency_ms=after, optimized=True)


TOGGLE = {"id": "tuning.cold_start_inference_auto", "default": True}
