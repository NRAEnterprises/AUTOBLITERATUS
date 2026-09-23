# abliterate/tuning/60_calibration_auto_tuning.py
#
# Tuning Type 60 of 103:
# Calibration auto tuning.
#
# Runtime toggle: --enable tuning.calibration_auto  (default: on)
#
# Effect:
#   Verify confidence tracks correctness, especially after ablation shifts
#   the refusal threshold.
#
# This file owns:
#   - calibration measurement (confidence-vs-correctness)
#   - recalibration

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class CalibrationProfile:
    auc: float
    corrected: bool


class CalibrationAutoTuning:

    def __init__(self, measure: Callable, apply_correction: Callable,
                 enabled: bool = True):
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


TOGGLE = {"id": "tuning.calibration_auto", "default": True}
