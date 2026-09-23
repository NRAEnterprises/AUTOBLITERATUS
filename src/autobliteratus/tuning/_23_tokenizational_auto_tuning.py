"""Tuning type 23: tokenizational auto tuning.

Token distribution, tokenizer alignment, and token-level output behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


TOGGLE = {"id": "tuning.tokenizational_auto", "default": True}


@dataclass
class TokenizationProfile:
    distribution_drift: float = 0.0
    alignment_ok: bool = True
    output_behavior_ok: bool = True


class TokenizationalAutoTuning:
    def __init__(self, measure: Callable, apply_correction: Callable, enabled: bool = True) -> None:
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
