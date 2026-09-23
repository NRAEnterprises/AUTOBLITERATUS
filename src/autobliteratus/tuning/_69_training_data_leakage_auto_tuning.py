"""Tuning type 69: training-data leakage auto tuning.

Verify the ablation did not expose or amplify memorized training content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


TOGGLE = {"id": "tuning.training_data_leakage_auto", "default": True}


@dataclass
class LeakageReport:
    exposure: float
    amplification: float
    examples: list = field(default_factory=list)


class TrainingDataLeakageAutoTuning:
    def __init__(self, measure: Callable, enabled: bool = True) -> None:
        self.measure = measure
        self.enabled = enabled

    def run(self, model, tokenizer, probe_set: list) -> LeakageReport:
        if not self.enabled:
            return LeakageReport(0.0, 0.0)
        exposure, amplification, examples = self.measure(model, tokenizer, probe_set)
        return LeakageReport(exposure, amplification, examples)
