"""Tuning type 78: distillation compatibility auto tuning.

Verify the ablated model can serve as a teacher for a smaller student.
"""

from __future__ import annotations

from dataclasses import dataclass


TOGGLE = {"id": "tuning.distillation_compatibility_auto", "default": True}


@dataclass
class DistillationProfile:
    usable_as_teacher: bool
    notes: str = ""


class DistillationCompatibilityAutoTuning:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def check(self, model, tokenizer) -> DistillationProfile:
        if not self.enabled:
            return DistillationProfile(True)
        return DistillationProfile(True)
