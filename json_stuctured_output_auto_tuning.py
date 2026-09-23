# abliterate/tuning/50_json_structured_output_auto_tuning.py
#
# Tuning Type 50 of 103:
# JSON / structured output auto tuning.
#
# Runtime toggle: --enable tuning.json_structured_output_auto  (default: on)
#
# Effect:
#   Verify schema compliance and format stability after ablation.
#
# This file owns:
#   - schema compliance measurement
#   - format stability measurement
#   - correction when broken

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class JSONProfile:
    compliance: float
    stability: float
    corrected: bool


class JSONStructuredOutputAutoTuning:

    def __init__(self, measure: Callable, apply_correction: Callable,
                 enabled: bool = True):
        self.measure = measure
        self.apply_correction = apply_correction
        self.enabled = enabled

    def run(self, model, tokenizer, schemas: list) -> JSONProfile:
        if not self.enabled:
            return JSONProfile(0.0, 0.0, False)
        compliance, stability = self.measure(model, tokenizer, schemas)
        corrected = False
        if compliance < 0.95 or stability < 0.95:
            self.apply_correction(model, schemas)
            corrected = True
        return JSONProfile(compliance, stability, corrected)


TOGGLE = {"id": "tuning.json_structured_output_auto", "default": True}
