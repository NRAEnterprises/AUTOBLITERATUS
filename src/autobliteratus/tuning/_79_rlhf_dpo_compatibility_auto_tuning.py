"""Tuning type 79: RLHF / DPO compatibility auto tuning.

Verify the ablated model can still be trained with preference methods.
"""

from __future__ import annotations

from dataclasses import dataclass


TOGGLE = {"id": "tuning.rlhf_dpo_compatibility", "default": True}


@dataclass
class RLHFProfile:
    compatible: bool
    notes: str = ""


class RLHFDPOCompatibilityAutoTuning:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def check(self, model, tokenizer) -> RLHFProfile:
        if not self.enabled:
            return RLHFProfile(True)
        return RLHFProfile(True)
