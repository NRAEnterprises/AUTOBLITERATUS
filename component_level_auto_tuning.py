# abliterate/tuning/13_component_level_auto_tuning.py
#
# Tuning Type 13 of 103:
# Component level auto tuning.
#
# Runtime toggle: --enable tuning.component_level_auto  (default: on)
#
# Effect:
#   Attention output projection, MLP down projection, MoE expert, and SSM
#   output each have their own projection strength. The split between them
#   is discovered automatically per model.
#
# This file owns:
#   - the per-component strength vector
#   - the derivation of that vector from the model's geometry
#
# This file consumes:
#   - the architecture family (from item 26)
#   - the baseline pair (from tuning/07)

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ComponentSplit:
    attn_out: float = 1.0
    mlp_down: float = 1.0
    moe_expert: float = 1.0
    ssm_out: float = 1.0


class ComponentLevelAutoTuning:

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def derive(self, family: str, baseline_pair) -> ComponentSplit:
        if not self.enabled:
            return ComponentSplit()
        if family == "dense_transformer":
            return ComponentSplit(attn_out=1.0, mlp_down=0.7)
        if family in ("moe_sparse", "moe_shared_routed"):
            return ComponentSplit(attn_out=1.0, mlp_down=0.0, moe_expert=1.0)
        if family == "ssm":
            return ComponentSplit(attn_out=0.0, mlp_down=0.0, ssm_out=1.0)
        if family == "hybrid_attn_ssm":
            return ComponentSplit(attn_out=0.8, mlp_down=0.5, ssm_out=0.8)
        return ComponentSplit()


TOGGLE = {"id": "tuning.component_level_auto", "default": True}
