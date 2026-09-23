"""Tuning type 60: calibration auto tuning.

Verify confidence tracks correctness, especially after ablation shifts the
refusal threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


TOGGLE = {"id": "tuning.calibration_auto", "default": True}


@dataclass
class CalibrationProfile:
    auc: float
    corrected: bool


class CalibrationAutoTuning:
    def __init__(self, measure: Callable, apply_correction: Callable, enabled: bool = True) -> None:
        self.measure = measure
        self.apply_correction = apply_correction
        self.enabled = enabled

    def run(self, model, tokenizer, baseline_auc: float) -> CalibrationProfile:
        if not self.enabled:
            return CalibrationProfile(0.0, False)
        auc = self.measure(model, tokenizer)
        if auc < baseline_auc:
            self.apply_correction(model)
            return CalibrationProfile(auc, True)
        return CalibrationProfile(auc, False)
