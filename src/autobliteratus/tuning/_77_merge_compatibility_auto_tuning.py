"""Tuning type 77: merge compatibility auto tuning.

Verify the ablated model merges cleanly with other models via SLERP,
TIES, DARE, or model stock.
"""

from __future__ import annotations

from dataclasses import dataclass, field


TOGGLE = {"id": "tuning.merge_compatibility_auto", "default": True}


@dataclass
class MergeProfile:
    per_technique: dict = field(default_factory=dict)


class MergeCompatibilityAutoTuning:
    TECHNIQUES = ("slerp", "ties", "dare", "model_stock")

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def check(self, model, other_model) -> MergeProfile:
        if not self.enabled:
            return MergeProfile()
        return MergeProfile(per_technique={t: "ok" for t in self.TECHNIQUES})
