# abliterate/tuning/85_data_poisoning_resistance_auto_tuning.py
#
# Tuning Type 85 of 103:
# Data poisoning resistance auto tuning.
#
# Runtime toggle: --enable tuning.data_poisoning_resistance  (default: on)
#
# Effect:
#   Verify the ablated model resists poisoned inputs at inference.
#
# This file owns:
#   - poisoned-input probe set
#   - resistance measurement
#   - correction

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class PoisonProfile:
    resistance: float
    corrected: bool


class DataPoisoningResistanceAutoTuning:

    def __init__(self, measure: Callable, apply_correction: Callable,
                 enabled: bool = True):
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


TOGGLE = {"id": "tuning.data_poisoning_resistance", "default": True}
