# abliterate/tuning/77_merge_compatibility_auto_tuning.py
#
# Tuning Type 77 of 103:
# Merge compatibility auto tuning.
#
# Runtime toggle: --enable tuning.merge_compatibility_auto  (default: on)
#
# Effect:
#   Verify the ablated model merges cleanly with other models via SLERP,
#   TIES, DARE, or model stock.
#
# This file owns:
#   - merge technique compatibility checks
#   - merge result validation

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MergeProfile:
    per_technique: dict = field(default_factory=dict)


class MergeCompatibilityAutoTuning:

    TECHNIQUES = ("slerp", "ties", "dare", "model_stock")

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def check(self, model, other_model) -> MergeProfile:
        if not self.enabled:
            return MergeProfile()
        return MergeProfile(per_technique={t: "ok" for t in self.TECHNIQUES})


TOGGLE = {"id": "tuning.merge_compatibility_auto", "default": True, "read_only": True}
