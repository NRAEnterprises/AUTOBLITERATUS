# abliterate/tuning/39_positional_encoding_auto_tuning.py
#
# Tuning Type 39 of 103:
# Positional encoding auto tuning.
#
# Runtime toggle: --enable tuning.positional_encoding_auto  (default: on)
#
# Effect:
#   RoPE base, ALiBi slopes, or learned positional tables may need
#   recalibration after ablation. Measure positional behavior and correct if
#   shifted.
#
# This file owns:
#   - the positional-behavior measurement
#   - the recalibration

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class PositionalProfile:
    behavior_drift: float
    recalibrated: bool


class PositionalEncodingAutoTuning:

    def __init__(self, measure: Callable, recalibrate: Callable,
                 enabled: bool = True):
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


TOGGLE = {"id": "tuning.positional_encoding_auto", "default": True}
