"""Tuning type 56: multimodal capability auto tuning.

For vision-language and audio-language models, restore vision and audio
pathways separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


TOGGLE = {"id": "tuning.multimodal_capability", "default": True}


@dataclass
class MultimodalPlan:
    per_modality_scores: dict = field(default_factory=dict)
    restored_modalities: list = field(default_factory=list)


class MultimodalCapabilityAutoTuning:
    MODALITIES = ("vision", "audio", "text")

    def __init__(self, measure: Callable, apply_correction: Callable, enabled: bool = True) -> None:
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
