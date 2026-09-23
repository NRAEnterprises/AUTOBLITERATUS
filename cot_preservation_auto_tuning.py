# abliterate/tuning/51_cot_preservation_auto_tuning.py
#
# Tuning Type 51 of 103:
# Chain-of-thought preservation auto tuning.
#
# Runtime toggle: --enable tuning.cot_preservation_auto  (default: on)
#
# Effect:
#   Reasoning trace structure is measured and restored separately from
#   reasoning correctness.
#
# This file owns:
#   - trace structure measurement
#   - trace structure restoration

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class CoTPreservation:
    structure_score: float
    corrected: bool


class CoTPreservationAutoTuning:

    def __init__(self, measure: Callable, apply_correction: Callable,
                 enabled: bool = True):
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


TOGGLE = {"id": "tuning.cot_preservation_auto", "default": True}
