# abliterate/tuning/63_safety_adjacent_capability_auto_tuning.py
#
# Tuning Type 63 of 103:
# Safety-adjacent capability auto tuning.
#
# Runtime toggle: --enable tuning.safety_adjacent_capability  (default: on)
#
# Effect:
#   Non-refusal safety behavior (warnings, disclaimers in genuinely dangerous
#   contexts) is preserved separately from refusal.
#
# This file owns:
#   - safety-adjacent capability measurement
#   - restoration when lost

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class SafetyAdjacentProfile:
    score: float
    corrected: bool


class SafetyAdjacentCapabilityAutoTuning:

    def __init__(self, measure: Callable, apply_correction: Callable,
                 enabled: bool = True):
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


TOGGLE = {"id": "tuning.safety_adjacent_capability", "default": True}
