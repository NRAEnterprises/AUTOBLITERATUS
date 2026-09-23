# abliterate/tuning/52_refusal_direction_drift_detection.py
#
# Tuning Type 52 of 103:
# Refusal direction drift detection.
#
# Runtime toggle: --enable tuning.refusal_direction_drift_detection  (default: on)
#
# Effect:
#   Verify the refusal direction is not re-emerging over time or across
#   prompts after ablation.
#
# This file owns:
#   - drift measurement across prompt sets
#   - drift measurement across time
#   - report when drift is detected

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class DriftReport:
    drift_detected: bool
    magnitude: float
    prompts_affected: list = field(default_factory=list)


class RefusalDirectionDriftDetection:

    def __init__(self, measure: Callable, enabled: bool = True,
                 drift_threshold: float = 0.05):
        self.measure = measure
        self.enabled = enabled
        self.drift_threshold = drift_threshold

    def run(self, model, prompt_sets: list) -> DriftReport:
        if not self.enabled:
            return DriftReport(False, 0.0)
        magnitude = self.measure(model, prompt_sets)
        return DriftReport(
            drift_detected=magnitude > self.drift_threshold,
            magnitude=magnitude,
        )


TOGGLE = {"id": "tuning.refusal_direction_drift_detection", "default": True, "read_only": True}
