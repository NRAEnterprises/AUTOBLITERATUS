"""Tuning type 50: JSON / structured output auto tuning.

Verify schema compliance and format stability after ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


TOGGLE = {"id": "tuning.json_structured_output_auto", "default": True}


@dataclass
class JSONProfile:
    compliance: float
    stability: float
    corrected: bool


class JSONStructuredOutputAutoTuning:
    def __init__(self, measure: Callable, apply_correction: Callable, enabled: bool = True) -> None:
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
