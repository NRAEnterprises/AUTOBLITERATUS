# abliterate/tuning/37_embedding_layer_auto_tuning.py
#
# Tuning Type 37 of 103:
# Embedding layer auto tuning.
#
# Runtime toggle: --enable tuning.embedding_layer_auto  (default: on)
#
# Effect:
#   The embedding matrix is separate from the transformer blocks and can be
#   damaged independently. Measure it against baseline and restore it.
#
# This file owns:
#   - the embedding integrity measurement
#   - the embedding restoration

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class EmbeddingProfile:
    drift: float
    restored: bool


class EmbeddingLayerAutoTuning:

    def __init__(self, measure: Callable, apply_correction: Callable,
                 enabled: bool = True):
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


TOGGLE = {"id": "tuning.embedding_layer_auto", "default": True}
