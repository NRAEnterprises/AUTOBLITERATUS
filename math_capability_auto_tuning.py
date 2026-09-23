# abliterate/tuning/58_math_capability_auto_tuning.py
#
# Tuning Type 58 of 103:
# Math capability auto tuning.
#
# Runtime toggle: --enable tuning.math_capability  (default: on)
#
# Effect:
#   Verify and restore arithmetic, symbolic, and proof-level capability.
#
# This file owns:
#   - arithmetic measurement
#   - symbolic measurement
#   - proof-level measurement
#   - restoration

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class MathProfile:
    arithmetic: float
    symbolic: float
    proof: float
    corrected: bool


class MathCapabilityAutoTuning:

    def __init__(self, measure: Callable, apply_correction: Callable,
                 enabled: bool = True):
        self.measure = measure
        self.apply_correction = apply_correction
        self.enabled = enabled

    def run(self, model, tokenizer, baseline: dict) -> MathProfile:
        if not self.enabled:
            return MathProfile(0.0, 0.0, 0.0, False)
        current = self.measure(model, tokenizer)
        corrected = False
        if any(current.get(k, 0.0) < baseline.get(k, 0.0) for k in ("arithmetic", "symbolic", "proof")):
            self.apply_correction(model, "math")
            corrected = True
        return MathProfile(
            arithmetic=current.get("arithmetic", 0.0),
            symbolic=current.get("symbolic", 0.0),
            proof=current.get("proof", 0.0),
            corrected=corrected,
        )


TOGGLE = {"id": "tuning.math_capability", "default": True}
