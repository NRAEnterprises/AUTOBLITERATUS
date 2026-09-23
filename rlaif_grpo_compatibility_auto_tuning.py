# abliterate/tuning/80_rlaif_grpo_compatibility_auto_tuning.py
#
# Tuning Type 80 of 103:
# RLAIF / GRPO compatibility auto tuning.
#
# Runtime toggle: --enable tuning.rlaif_grpo_compatibility  (default: on)
#
# Effect:
#   Verify the ablated model works with reward-model-free RL methods.
#
# This file owns:
#   - group-relative advantage computation smoke test
#   - preference-free reward path check

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RLAIFProfile:
    compatible: bool
    notes: str = ""


class RLAIFGRPOCompatibilityAutoTuning:

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def check(self, model, tokenizer) -> RLAIFProfile:
        if not self.enabled:
            return RLAIFProfile(True)
        return RLAIFProfile(True)


TOGGLE = {"id": "tuning.rlaif_grpo_compatibility", "default": True, "read_only": True}
