"""Tuning type 51: chain-of-thought preservation auto tuning.

Reasoning trace structure is measured and restored separately from
reasoning correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


TOGGLE = {"id": "tuning.cot_preservation_auto", "default": True}


@dataclass
class CoTPreservation:
    structure_score: float
    corrected: bool


class CoTPreservationAutoTuning:
    def __init__(self, measure: Callable, apply_correction: Callable, enabled: bool = True) -> None:
        self.measure = measure
        self.apply_correction = apply_correction
        self.enabled = enabled

    def run(self, model, tokenizer, baseline: float) -> CoTPreservation:
        if not self.enabled:
            return CoTPreservation(0.0, False)
        current = self.measure(model, tokenizer)
        if current < baseline:
            self.apply_correction(model)
            return CoTPreservation(current, True)
        return CoTPreservation(current, False)
