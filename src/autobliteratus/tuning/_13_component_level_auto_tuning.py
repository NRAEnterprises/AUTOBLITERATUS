"""Tuning type 13: component level auto tuning.

Attention, MLP, MoE expert, SSM output each get their own projection
strength, derived per model.
"""

from __future__ import annotations

from dataclasses import dataclass


TOGGLE = {"id": "tuning.component_level_auto", "default": True}


@dataclass
class ComponentSplit:
    attn_out: float = 1.0
    mlp_down: float = 1.0
    moe_expert: float = 1.0
    ssm_out: float = 1.0


class ComponentLevelAutoTuning:
    def __init__(self, enabled: bool = True) -> None:
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
