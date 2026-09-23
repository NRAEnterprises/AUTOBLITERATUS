"""Tuning type 42: context window auto tuning.

The effective context length may shrink after ablation. Measure it and
compensate.
"""

from __future__ import annotations

from dataclasses import dataclass


TOGGLE = {"id": "tuning.context_window_auto", "default": True}


@dataclass
class ContextPlan:
    effective_length: int
    original_length: int
    compensated: bool


class ContextWindowAutoTuning:
    def __init__(self, measure, compensate, enabled: bool = True) -> None:
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
