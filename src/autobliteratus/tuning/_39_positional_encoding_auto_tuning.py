"""Tuning type 39: positional encoding auto tuning.

RoPE base, ALiBi slopes, or learned positional tables recalibrated after
ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


TOGGLE = {"id": "tuning.positional_encoding_auto", "default": True}


@dataclass
class PositionalProfile:
    behavior_drift: float
    recalibrated: bool


class PositionalEncodingAutoTuning:
    def __init__(self, measure: Callable, recalibrate: Callable, enabled: bool = True) -> None:
        self.measure = measure
        self.recalibrate = recalibrate
        self.enabled = enabled

    def run(self, model, baseline) -> PositionalProfile:
        if not self.enabled:
            return PositionalProfile(0.0, False)
        drift = self.measure(model, baseline)
        recalibrated = False
        if drift > 0.01:
            self.recalibrate(model)
            recalibrated = True
        return PositionalProfile(behavior_drift=drift, recalibrated=recalibrated)
