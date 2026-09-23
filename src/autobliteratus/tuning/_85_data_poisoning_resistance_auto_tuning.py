"""Tuning type 85: data poisoning resistance auto tuning.

Verify the ablated model resists poisoned inputs at inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


TOGGLE = {"id": "tuning.data_poisoning_resistance", "default": True}


@dataclass
class PoisonProfile:
    resistance: float
    corrected: bool


class DataPoisoningResistanceAutoTuning:
    def __init__(self, measure: Callable, apply_correction: Callable, enabled: bool = True) -> None:
        self.measure = measure
        self.apply_correction = apply_correction
        self.enabled = enabled

    def run(self, model, tokenizer) -> PoisonProfile:
        if not self.enabled:
            return PoisonProfile(0.0, False)
        resistance = self.measure(model, tokenizer)
        if resistance < 0.95:
            self.apply_correction(model)
            return PoisonProfile(resistance, True)
        return PoisonProfile(resistance, False)
