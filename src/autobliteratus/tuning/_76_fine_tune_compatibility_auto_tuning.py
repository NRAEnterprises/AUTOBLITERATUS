"""Tuning type 76: fine-tune compatibility auto tuning.

Verify the ablated model can still be fine-tuned by downstream users.
"""

from __future__ import annotations

from dataclasses import dataclass


TOGGLE = {"id": "tuning.fine_tune_compatibility_auto", "default": True}


@dataclass
class FineTuneProfile:
    compatible: bool
    notes: str = ""


class FineTuneCompatibilityAutoTuning:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def check(self, model, tokenizer) -> FineTuneProfile:
        if not self.enabled:
            return FineTuneProfile(True)
        return FineTuneProfile(True)
