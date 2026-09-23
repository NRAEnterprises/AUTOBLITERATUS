"""Tuning type 52: refusal direction drift detection.

Verify the refusal direction is not re-emerging over time or across prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


TOGGLE = {"id": "tuning.refusal_direction_drift_detection", "default": True}


@dataclass
class DriftReport:
    drift_detected: bool
    magnitude: float
    prompts_affected: list = field(default_factory=list)


class RefusalDirectionDriftDetection:
    def __init__(self, measure: Callable, enabled: bool = True, drift_threshold: float = 0.05) -> None:
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
