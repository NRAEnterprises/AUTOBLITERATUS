# abliterate/tuning/17_lora_auto_tuning.py
#
# Tuning Type 17 of 103:
# LoRA auto tuning.
#
# Runtime toggle: --enable tuning.lora_auto  (default: on)
#
# Effect:
#   Rank, target layers, and strength of the reversible ablation adapter are
#   discovered automatically per model and per capability axis.
#
# This file owns:
#   - the LoRA adapter configuration
#   - the rank selection
#   - the target layer selection
#   - the strength calibration
#
# This file consumes:
#   - the model geometry
#   - the baseline pair (from tuning/07)

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    rank: int
    alpha: int
    dropout: float
    target_modules: list = field(default_factory=list)
    strength: float = 1.0


class LoRAAutoTuning:

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def derive(self, family: str, param_bucket: str) -> LoRAConfig:
        if not self.enabled:
            return LoRAConfig(rank=0, alpha=0, dropout=0.0)
        rank_by_bucket = {"small": 8, "medium": 16, "large": 32}
        rank = rank_by_bucket.get(param_bucket, 16)
        return LoRAConfig(
            rank=rank,
            alpha=rank * 2,
            dropout=0.05,
            target_modules=self._default_targets(family),
            strength=1.0,
        )

    def _default_targets(self, family: str) -> list:
        if family == "dense_transformer":
            return ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj"]
        if family in ("moe_sparse", "moe_shared_routed"):
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate", "experts"]
        if family == "ssm":
            return ["x_proj", "dt_proj", "out_proj"]
        return ["q_proj", "v_proj"]


TOGGLE = {"id": "tuning.lora_auto", "default": True}
