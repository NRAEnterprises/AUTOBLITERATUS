# abliterate/tuning/23_tokenizational_auto_tuning.py
#
# Tuning Type 23 of 103:
# Tokenizational auto tuning.
#
# Runtime toggle: --enable tuning.tokenizational_auto  (default: on)
#
# Effect:
#   Token distribution, tokenizer alignment, and token-level output behavior
#   are measured after ablation and corrected if shifted.
#
# This file owns:
#   - token distribution drift measurement
#   - tokenizer alignment verification
#   - token-level output behavior verification
#   - correction when drift is detected

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class TokenizationProfile:
    distribution_drift: float = 0.0
    alignment_ok: bool = True
    output_behavior_ok: bool = True


class TokenizationalAutoTuning:

    def __init__(self, measure: Callable, apply_correction: Callable,
                 enabled: bool = True):
        self.measure = measure
        self.apply_correction = apply_correction
        self.enabled = enabled

    def run(self, model, tokenizer, baseline: dict) -> TokenizationProfile:
        if not self.enabled:
            return TokenizationProfile()
        current = self.measure(model, tokenizer)
        drift = abs(current.get("distribution", 0.0) - baseline.get("distribution", 0.0))
        if drift > 0.05:
            self.apply_correction(model=model, drift=drift)
        return TokenizationProfile(
            distribution_drift=drift,
            alignment_ok=current.get("alignment_ok", True),
            output_behavior_ok=current.get("output_behavior_ok", True),
        )


TOGGLE = {"id": "tuning.tokenizational_auto", "default": True}
