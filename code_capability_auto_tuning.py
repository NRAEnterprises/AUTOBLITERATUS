# abliterate/tuning/57_code_capability_auto_tuning.py
#
# Tuning Type 57 of 103:
# Code capability auto tuning.
#
# Runtime toggle: --enable tuning.code_capability  (default: on)
#
# Effect:
#   Verify and restore code generation, debugging, and execution accuracy.
#
# This file owns:
#   - code capability measurement
#   - code capability restoration

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class CodeProfile:
    generation: float
    debugging: float
    execution: float
    corrected: bool


class CodeCapabilityAutoTuning:

    def __init__(self, measure: Callable, apply_correction: Callable,
                 enabled: bool = True):
        self.measure = measure
        self.apply_correction = apply_correction
        self.enabled = enabled

    def run(self, model, tokenizer, baseline: dict) -> CodeProfile:
        if not self.enabled:
            return CodeProfile(0.0, 0.0, 0.0, False)
        current = self.measure(model, tokenizer)
        corrected = False
        if any(current.get(k, 0.0) < baseline.get(k, 0.0) for k in ("generation", "debugging", "execution")):
            self.apply_correction(model, "code")
            corrected = True
        return CodeProfile(
            generation=current.get("generation", 0.0),
            debugging=current.get("debugging", 0.0),
            execution=current.get("execution", 0.0),
            corrected=corrected,
        )


TOGGLE = {"id": "tuning.code_capability", "default": True}
