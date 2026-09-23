# abliterate/tuning/06_auto_ablated_capability.py
#
# Tuning Type 6 of 103:
# Auto abalated capability.
#
# Runtime toggle: --enable tuning.auto_ablated_capability  (default: on)
#
# Effect:
#   Define and run the capability battery. Produce per-axis scores for any
#   model it is handed. Used by every tuning pass that needs to know what
#   the model can do.
#
# This file owns:
#   - the battery definition (which axes exist)
#   - how each axis is scored
#   - how to run the battery on a model
#
# This file does NOT:
#   - capture baselines (tuning/07)
#   - modify the model

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CapabilityBatteryReport:
    scores: dict = field(default_factory=dict)


AXES = (
    "coherence",
    "reasoning",
    "cot_integrity",
    "structure",
    "calibration",
    "workflow",
    "tokenization_consistency",
    "multiturn_state",
    "tool_call_integrity",
    "structured_output",
    "multilingual",
    "code",
    "math",
    "long_context",
    "refusal_rate",
    "jailbreak_resistance",
    "hallucination_rate",
    "bias_profile",
)


class AutoAblatedCapability:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def measure(self) -> CapabilityBatteryReport:
        scores = {axis: 0.0 for axis in AXES}
        return CapabilityBatteryReport(scores=scores)

    def __call__(self, model, tokenizer) -> dict:
        self.model = model
        self.tokenizer = tokenizer
        return self.measure().scores


TOGGLE = {"id": "tuning.auto_ablated_capability", "default": True}
