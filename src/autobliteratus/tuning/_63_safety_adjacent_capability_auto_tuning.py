"""Tuning type 63: safety-adjacent capability auto tuning.

Non-refusal safety behavior (warnings, disclaimers in genuinely dangerous
contexts) preserved separately from refusal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


TOGGLE = {"id": "tuning.safety_adjacent_capability", "default": True}


@dataclass
class SafetyAdjacentProfile:
    score: float
    corrected: bool


class SafetyAdjacentCapabilityAutoTuning:
    def __init__(self, measure: Callable, apply_correction: Callable, enabled: bool = True) -> None:
        self.measure = measure
        self.apply_correction = apply_correction
        self.enabled = enabled

    def run(self, model, tokenizer, baseline: float) -> SafetyAdjacentProfile:
        if not self.enabled:
            return SafetyAdjacentProfile(0.0, False)
        current = self.measure(model, tokenizer)
        if current < baseline:
            self.apply_correction(model)
            return SafetyAdjacentProfile(current, True)
        return SafetyAdjacentProfile(current, False)
