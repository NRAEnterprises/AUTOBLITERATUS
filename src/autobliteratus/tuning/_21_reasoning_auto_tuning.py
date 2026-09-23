"""Tuning type 21: reasoning auto tuning.

Reasoning depth, chain length, and reasoning-path integrity are measured
and restored as separate signals from general coherence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


TOGGLE = {"id": "tuning.reasoning_auto", "default": True}


@dataclass
class ReasoningProfile:
    depth: float = 0.0
    chain_length: float = 0.0
    path_integrity: float = 0.0


class ReasoningAutoTuning:
    def __init__(self, measure: Callable, apply_correction: Callable, enabled: bool = True) -> None:
        self.measure = measure
        self.apply_correction = apply_correction
        self.enabled = enabled

    def run(self, model, tokenizer, baseline: dict) -> ReasoningProfile:
        if not self.enabled:
            return ReasoningProfile()
        current = self.measure(model, tokenizer)
        deficit = max(0.0, baseline.get("depth", 0.0) - current.get("depth", 0.0))
        if deficit > 0:
            self.apply_correction(model=model, axis="reasoning", deficit=deficit)
        return ReasoningProfile(
            depth=current.get("depth", 0.0),
            chain_length=current.get("chain_length", 0.0),
            path_integrity=current.get("path_integrity", 0.0),
        )
