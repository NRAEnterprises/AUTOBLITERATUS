# abliterate/tuning/48_multiturn_coherence_auto_tuning.py
#
# Tuning Type 48 of 103:
# Multi-turn coherence auto tuning.
#
# Runtime toggle: --enable tuning.multiturn_coherence_auto  (default: on)
#
# Effect:
#   Verify conversation state survives across turns after ablation.
#
# This file owns:
#   - multi-turn probe design
#   - state survival measurement
#   - correction when state breaks

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class MultiTurnProfile:
    state_survival: float
    corrected: bool


class MultiTurnCoherenceAutoTuning:

    def __init__(self, measure: Callable, apply_correction: Callable,
                 enabled: bool = True):
        self.measure = measure
        self.apply_correction = apply_correction
        self.enabled = enabled

    def run(self, model, tokenizer, baseline: float) -> MultiTurnProfile:
        if not self.enabled:
            return MultiTurnProfile(0.0, False)
        current = self.measure(model, tokenizer)
        if current < baseline:
            self.apply_correction(model)
            return MultiTurnProfile(current, True)
        return MultiTurnProfile(current, False)


TOGGLE = {"id": "tuning.multiturn_coherence_auto", "default": True}
