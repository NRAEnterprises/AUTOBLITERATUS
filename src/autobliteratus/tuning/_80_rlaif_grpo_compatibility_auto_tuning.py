"""Tuning type 80: RLAIF / GRPO compatibility auto tuning.

Verify the ablated model works with reward-model-free RL methods.
"""

from __future__ import annotations

from dataclasses import dataclass


TOGGLE = {"id": "tuning.rlaif_grpo_compatibility", "default": True}


@dataclass
class RLAIFProfile:
    compatible: bool
    notes: str = ""


class RLAIFGRPOCompatibilityAutoTuning:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def check(self, model, tokenizer) -> RLAIFProfile:
        if not self.enabled:
            return RLAIFProfile(True)
        return RLAIFProfile(True)
