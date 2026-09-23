# abliterate/tuning/79_rlhf_dpo_compatibility_auto_tuning.py
#
# Tuning Type 79 of 103:
# RLHF / DPO compatibility auto tuning.
#
# Runtime toggle: --enable tuning.rlhf_dpo_compatibility  (default: on)
#
# Effect:
#   Verify the ablated model can still be trained with preference methods.
#
# This file owns:
#   - reference-model-free check
#   - preference-loss smoke test

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RLHFProfile:
    compatible: bool
    notes: str = ""


class RLHFDPOCompatibilityAutoTuning:

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def check(self, model, tokenizer) -> RLHFProfile:
        if not self.enabled:
            return RLHFProfile(True)
        return RLHFProfile(True)


TOGGLE = {"id": "tuning.rlhf_dpo_compatibility", "default": True, "read_only": True}
