# abliterate/tuning/18_moe_auto_tuning.py
#
# Tuning Type 18 of 103:
# MoE auto tuning.
#
# Runtime toggle: --enable tuning.moe_auto  (default: on)
#
# Effect:
#   Expert selection, per-expert refusal direction, and inter-expert
#   compensation for MoE models.
#
# This file owns:
#   - which experts are involved
#   - per-expert direction computation
#   - compensation across experts so no single expert carries the load
#
# This file consumes:
#   - the MoE model itself
#   - the refusal probe (from PROBE)

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MoEPlan:
    expert_indices: list = field(default_factory=list)
    expert_directions: dict = field(default_factory=dict)
    compensation_matrix: dict = field(default_factory=dict)


class MoEAutoTuning:

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def derive(self, model, refusal_probe: dict) -> MoEPlan:
        if not self.enabled:
            return MoEPlan()
        return MoEPlan()


TOGGLE = {"id": "tuning.moe_auto", "default": True}
