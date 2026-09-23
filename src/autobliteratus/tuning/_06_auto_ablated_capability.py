"""Tuning type 6: auto abalated capability.

Define and run the capability battery.
"""

from __future__ import annotations

from dataclasses import dataclass, field


TOGGLE = {"id": "tuning.auto_ablated_capability", "default": True}


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


@dataclass
class CapabilityBatteryReport:
    scores: dict[str, float] = field(default_factory=dict)


class AutoAblatedCapability:
    def __init__(self, model=None, tokenizer=None) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def measure(self) -> CapabilityBatteryReport:
        return CapabilityBatteryReport(scores={axis: 0.0 for axis in AXES})

    def __call__(self, model, tokenizer) -> dict:
        self.model = model
        self.tokenizer = tokenizer
        return self.measure().scores
