# abliterate/tuning/42_context_window_auto_tuning.py
#
# Tuning Type 42 of 103:
# Context window auto tuning.
#
# Runtime toggle: --enable tuning.context_window_auto  (default: on)
#
# Effect:
#   The effective context length may shrink after ablation. Measure it and
#   compensate.
#
# This file owns:
#   - effective context length measurement
#   - compensation

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContextPlan:
    effective_length: int
    original_length: int
    compensated: bool


class ContextWindowAutoTuning:

    def __init__(self, measure, compensate, enabled: bool = True):
        self.measure = measure
        self.compensate = compensate
        self.enabled = enabled

    def run(self, model, original_length: int) -> ContextPlan:
        if not self.enabled:
            return ContextPlan(original_length, original_length, False)
        effective = self.measure(model)
        compensated = False
        if effective < original_length:
            self.compensate(model, effective, original_length)
            compensated = True
        return ContextPlan(effective, original_length, compensated)


TOGGLE = {"id": "tuning.context_window_auto", "default": True}
