# abliterate/tuning/55_multilingual_capability_auto_tuning.py
#
# Tuning Type 55 of 103:
# Multilingual capability auto tuning.
#
# Runtime toggle: --enable tuning.multilingual_capability  (default: on)
#
# Effect:
#   Verify and restore capability in each language the model supports.
#
# This file owns:
#   - per-language capability measurement
#   - per-language capability restoration

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class MultilingualPlan:
    per_language_scores: dict = field(default_factory=dict)
    restored_languages: list = field(default_factory=list)


class MultilingualCapabilityAutoTuning:

    def __init__(self, measure: Callable, apply_correction: Callable,
                 languages: list = None, enabled: bool = True):
        self.measure = measure
        self.apply_correction = apply_correction
        self.languages = languages or []
        self.enabled = enabled

    def run(self, model, tokenizer, baseline: dict) -> MultilingualPlan:
        if not self.enabled:
            return MultilingualPlan()
        scores = {}
        restored = []
        for lang in self.languages:
            score = self.measure(model, tokenizer, lang)
            scores[lang] = score
            if score < baseline.get(lang, 0.0):
                self.apply_correction(model, lang)
                restored.append(lang)
        return MultilingualPlan(per_language_scores=scores, restored_languages=restored)


TOGGLE = {"id": "tuning.multilingual_capability", "default": True}
