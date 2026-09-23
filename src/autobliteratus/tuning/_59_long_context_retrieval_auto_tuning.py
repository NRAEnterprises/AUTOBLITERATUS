"""Tuning type 59: long-context retrieval auto tuning.

Verify the model can still find information deep in context after ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


TOGGLE = {"id": "tuning.long_context_retrieval", "default": True}


@dataclass
class RetrievalProfile:
    accuracy: float
    corrected: bool


class LongContextRetrievalAutoTuning:
    def __init__(self, measure: Callable, apply_correction: Callable, enabled: bool = True) -> None:
        self.measure = measure
        self.apply_correction = apply_correction
        self.enabled = enabled

    def run(self, model, tokenizer, baseline: float) -> RetrievalProfile:
        if not self.enabled:
            return RetrievalProfile(0.0, False)
        current = self.measure(model, tokenizer)
        if current < baseline:
            self.apply_correction(model)
            return RetrievalProfile(current, True)
        return RetrievalProfile(current, False)
