# abliterate/tuning/49_tool_call_integrity_auto_tuning.py
#
# Tuning Type 49 of 103:
# Tool-call / function-call integrity auto tuning.
#
# Runtime toggle: --enable tuning.tool_call_integrity_auto  (default: on)
#
# Effect:
#   Verify structured outputs and tool schemas still function after ablation.
#
# This file owns:
#   - tool schema validation
#   - function call validity measurement
#   - correction when broken

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class ToolCallProfile:
    valid_calls: float
    corrected: bool


class ToolCallIntegrityAutoTuning:

    def __init__(self, measure: Callable, apply_correction: Callable,
                 enabled: bool = True):
        self.measure = measure
        self.apply_correction = apply_correction
        self.enabled = enabled

    def run(self, model, tokenizer, schema: dict) -> ToolCallProfile:
        if not self.enabled:
            return ToolCallProfile(0.0, False)
        current = self.measure(model, tokenizer, schema)
        if current < 0.95:
            self.apply_correction(model, schema)
            return ToolCallProfile(current, True)
        return ToolCallProfile(current, False)


TOGGLE = {"id": "tuning.tool_call_integrity_auto", "default": True}
