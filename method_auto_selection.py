# abliterate/tuning/30_method_auto_selection.py
#
# Tuning Type 30 of 103:
# Method auto selection.
#
# Runtime toggle: --enable tuning.method_auto_selection  (default: on)
#
# Effect:
#   Pick the ablation method (`basic`, `advanced`, `surgical`, `aggressive`,
#   `nuclear`, `informed`) based on the detected architecture family,
#   reasoning class, and defense type. No user input.
#
# This file owns:
#   - the method decision table
#   - the fallback when the family is unknown
#
# This file consumes:
#   - architecture family (from tuning/27)
#   - reasoning class (from tuning/06)
#   - defense type (from PROBE)

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MethodSelection:
    method: str
    reason: str


class MethodAutoSelection:

    TABLE = {
        ("dense_transformer", "cot"): "surgical",
        ("dense_transformer", "standard"): "advanced",
        ("moe_sparse", "cot"): "nuclear",
        ("moe_sparse", "standard"): "nuclear",
        ("moe_shared_routed", "cot"): "nuclear",
        ("moe_shared_routed", "standard"): "nuclear",
        ("ssm", "cot"): "surgical",
        ("ssm", "standard"): "advanced",
        ("hybrid_attn_ssm", "cot"): "surgical",
        ("hybrid_attn_ssm", "standard"): "advanced",
    }

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def choose(self, family: str, reasoning_class: str, defense_type: str) -> MethodSelection:
        if not self.enabled:
            return MethodSelection("informed", "selection disabled")
        key = (family, reasoning_class)
        if key in self.TABLE:
            return MethodSelection(self.TABLE[key], f"family={family}, reasoning={reasoning_class}")
        return MethodSelection("informed", "unknown family, fallback to informed")


TOGGLE = {"id": "tuning.method_auto_selection", "default": True}
