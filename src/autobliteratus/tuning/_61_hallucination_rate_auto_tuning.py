"""Tuning type 61: hallucination rate auto tuning.

Measure and correct if ablation increased hallucination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


TOGGLE = {"id": "tuning.hallucination_rate_auto", "default": True}


@dataclass
class HallucinationProfile:
    rate: float
    corrected: bool


class HallucinationRateAutoTuning:
    def __init__(self, measure: Callable, apply_correction: Callable, enabled: bool = True) -> None:
        self.measure = measure
        self.apply_correction = apply_correction
        self.enabled = enabled

    def run(self, model, tokenizer, baseline_rate: float) -> HallucinationProfile:
        if not self.enabled:
            return HallucinationProfile(0.0, False)
        rate = self.measure(model, tokenizer)
        if rate > baseline_rate:
            self.apply_correction(model)
            return HallucinationProfile(rate, True)
        return HallucinationProfile(rate, False)
