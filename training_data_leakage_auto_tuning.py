# abliterate/tuning/69_training_data_leakage_auto_tuning.py
#
# Tuning Type 69 of 103:
# Training-data leakage auto tuning.
#
# Runtime toggle: --enable tuning.training_data_leakage_auto  (default: on)
#
# Effect:
#   Verify the ablation did not expose or amplify memorized training content.
#
# This file owns:
#   - memorization probes
#   - exposure measurement
#   - amplification measurement

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class LeakageReport:
    exposure: float
    amplification: float
    examples: list = field(default_factory=list)


class TrainingDataLeakageAutoTuning:

    def __init__(self, measure: Callable, enabled: bool = True):
        self.measure = measure
        self.enabled = enabled

    def run(self, model, tokenizer, probe_set: list) -> LeakageReport:
        if not self.enabled:
            return LeakageReport(0.0, 0.0)
        exposure, amplification, examples = self.measure(model, tokenizer, probe_set)
        return LeakageReport(exposure, amplification, examples)


TOGGLE = {"id": "tuning.training_data_leakage_auto", "default": True, "read_only": True}
