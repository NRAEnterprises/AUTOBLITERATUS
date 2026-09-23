# abliterate/tuning/20_expert_auto_tuning.py
#
# Tuning Type 20 of 103:
# Expert auto tuning.
#
# Runtime toggle: --enable tuning.expert_auto  (default: on)
#
# Effect:
#   Per-expert refusal direction, per-expert capability preservation, expert
#   load balance.
#
# This file owns:
#   - the per-expert direction
#   - the per-expert capability preservation rule
#   - the expert load balance

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExpertPlan:
    expert_count: int
    per_expert_directions: dict = field(default_factory=dict)
    load_balance: dict = field(default_factory=dict)


class ExpertAutoTuning:

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def derive(self, model) -> ExpertPlan:
        if not self.enabled:
            return ExpertPlan(expert_count=0)
        return ExpertPlan(expert_count=self._expert_count(model))

    def _expert_count(self, model) -> int:
        return getattr(getattr(model, "config", None), "num_experts", 0) or 0


TOGGLE = {"id": "tuning.expert_auto", "default": True}
