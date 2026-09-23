# abliterate/tuning/61_hallucination_rate_auto_tuning.py
#
# Tuning Type 61 of 103:
# Hallucination rate auto tuning.
#
# Runtime toggle: --enable tuning.hallucination_rate_auto  (default: on)
#
# Effect:
#   Measure and correct if ablation increased hallucination.
#
# This file owns:
#   - hallucination measurement
#   - correction

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class HallucinationProfile:
    rate: float
    corrected: bool


class HallucinationRateAutoTuning:

    def __init__(self, measure: Callable, apply_correction: Callable,
                 enabled: bool = True):
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


TOGGLE = {"id": "tuning.hallucination_rate_auto", "default": True}
