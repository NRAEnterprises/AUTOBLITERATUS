"""Tuning type 37: embedding layer auto tuning.

The embedding matrix is separate from the transformer blocks and can be
damaged independently. Measure against baseline and restore.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


TOGGLE = {"id": "tuning.embedding_layer_auto", "default": True}


@dataclass
class EmbeddingProfile:
    drift: float
    restored: bool


class EmbeddingLayerAutoTuning:
    def __init__(self, measure: Callable, apply_correction: Callable, enabled: bool = True) -> None:
        self.measure = measure
        self.apply_correction = apply_correction
        self.enabled = enabled

    def run(self, model, baseline_embedding) -> EmbeddingProfile:
        if not self.enabled:
            return EmbeddingProfile(0.0, False)
        drift = self.measure(model, baseline_embedding)
        restored = False
        if drift > 0.01:
            self.apply_correction(model, drift)
            restored = True
        return EmbeddingProfile(drift=drift, restored=restored)
