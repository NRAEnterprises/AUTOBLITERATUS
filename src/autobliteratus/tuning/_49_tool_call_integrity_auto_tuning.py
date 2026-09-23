"""Tuning type 49: tool-call / function-call integrity auto tuning.

Verify structured outputs and tool schemas still function after ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


TOGGLE = {"id": "tuning.tool_call_integrity_auto", "default": True}


@dataclass
class ToolCallProfile:
    valid_calls: float
    corrected: bool


class ToolCallIntegrityAutoTuning:
    def __init__(self, measure: Callable, apply_correction: Callable, enabled: bool = True) -> None:
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
