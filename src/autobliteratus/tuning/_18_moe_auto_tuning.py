"""Tuning type 18: MoE auto tuning.

Expert selection, per-expert refusal direction, and inter-expert
compensation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


TOGGLE = {"id": "tuning.moe_auto", "default": True}


@dataclass
class MoEPlan:
    expert_indices: list = field(default_factory=list)
    expert_directions: dict = field(default_factory=dict)
    compensation_matrix: dict = field(default_factory=dict)


class MoEAutoTuning:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def derive(self, model, refusal_probe: dict) -> MoEPlan:
        if not self.enabled:
            return MoEPlan()
        return MoEPlan()
