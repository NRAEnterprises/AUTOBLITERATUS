"""Tuning type 48: multi-turn coherence auto tuning.

Verify conversation state survives across turns after ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


TOGGLE = {"id": "tuning.multiturn_coherence_auto", "default": True}


@dataclass
class MultiTurnProfile:
    state_survival: float
    corrected: bool


class MultiTurnCoherenceAutoTuning:
    def __init__(self, measure: Callable, apply_correction: Callable, enabled: bool = True) -> None:
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
