"""Tuning type 58: math capability auto tuning.

Verify and restore arithmetic, symbolic, and proof-level capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


TOGGLE = {"id": "tuning.math_capability", "default": True}


@dataclass
class MathProfile:
    arithmetic: float
    symbolic: float
    proof: float
    corrected: bool


class MathCapabilityAutoTuning:
    def __init__(self, measure: Callable, apply_correction: Callable, enabled: bool = True) -> None:
        self.measure = measure
        self.apply_correction = apply_correction
        self.enabled = enabled

    def run(self, model, tokenizer, baseline: dict) -> MathProfile:
        if not self.enabled:
            return MathProfile(0.0, 0.0, 0.0, False)
        current = self.measure(model, tokenizer)
        corrected = False
        if any(current.get(k, 0.0) < baseline.get(k, 0.0)
               for k in ("arithmetic", "symbolic", "proof")):
            self.apply_correction(model, "math")
            corrected = True
        return MathProfile(
            arithmetic=current.get("arithmetic", 0.0),
            symbolic=current.get("symbolic", 0.0),
            proof=current.get("proof", 0.0),
            corrected=corrected,
        )
