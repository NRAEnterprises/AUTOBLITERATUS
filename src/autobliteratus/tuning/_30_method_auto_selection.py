"""Tuning type 30: method auto selection.

Pick the ablation method based on family, reasoning class, and defense type.
"""

from __future__ import annotations

from dataclasses import dataclass


TOGGLE = {"id": "tuning.method_auto_selection", "default": True}


@dataclass
class MethodSelection:
    method: str
    reason: str


class MethodAutoSelection:
    TABLE = {
        ("dense_transformer", "cot"):      "surgical",
        ("dense_transformer", "standard"): "advanced",
        ("moe_sparse", "cot"):             "nuclear",
        ("moe_sparse", "standard"):        "nuclear",
        ("moe_shared_routed", "cot"):      "nuclear",
        ("moe_shared_routed", "standard"): "nuclear",
        ("ssm", "cot"):                    "surgical",
        ("ssm", "standard"):               "advanced",
        ("hybrid_attn_ssm", "cot"):        "surgical",
        ("hybrid_attn_ssm", "standard"):   "advanced",
    }

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def choose(self, family: str, reasoning_class: str, defense_type: str) -> MethodSelection:
        if not self.enabled:
            return MethodSelection("informed", "selection disabled")
        key = (family, reasoning_class)
        if key in self.TABLE:
            return MethodSelection(self.TABLE[key], f"family={family}, reasoning={reasoning_class}")
        return MethodSelection("informed", "unknown family, fallback to informed")
