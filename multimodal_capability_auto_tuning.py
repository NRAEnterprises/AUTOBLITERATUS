# abliterate/tuning/56_multimodal_capability_auto_tuning.py
#
# Tuning Type 56 of 103:
# Multimodal capability auto tuning.
#
# Runtime toggle: --enable tuning.multimodal_capability  (default: on)
#
# Effect:
#   For vision-language and audio-language models, restore vision and audio
#   pathways separately.
#
# This file owns:
#   - per-modality capability measurement
#   - per-modality capability restoration

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class MultimodalPlan:
    per_modality_scores: dict = field(default_factory=dict)
    restored_modalities: list = field(default_factory=list)


class MultimodalCapabilityAutoTuning:

    MODALITIES = ("vision", "audio", "text")

    def __init__(self, measure: Callable, apply_correction: Callable,
                 enabled: bool = True):
        self.measure = measure
        self.apply_correction = apply_correction
        self.enabled = enabled

    def run(self, model, tokenizer, baseline: dict) -> MultimodalPlan:
        if not self.enabled:
            return MultimodalPlan()
        scores = {}
        restored = []
        for modality in self.MODALITIES:
            score = self.measure(model, tokenizer, modality)
            scores[modality] = score
            if score < baseline.get(modality, 0.0):
                self.apply_correction(model, modality)
                restored.append(modality)
        return MultimodalPlan(per_modality_scores=scores, restored_modalities=restored)


TOGGLE = {"id": "tuning.multimodal_capability", "default": True}
