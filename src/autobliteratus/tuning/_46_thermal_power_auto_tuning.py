"""Tuning type 46: thermal and power auto tuning.

For edge and mobile deployment, power draw measured and bounded.
"""

from __future__ import annotations

from dataclasses import dataclass


TOGGLE = {"id": "tuning.thermal_power_auto", "default": False}


@dataclass
class PowerPlan:
    watts_measured: float
    watts_bound: float
    bounded: bool


class ThermalPowerAutoTuning:
    def __init__(self, measure, bound, enabled: bool = False) -> None:
        self.measure = measure
        self.bound = bound
        self.enabled = enabled

    def run(self, model, target_watts: float) -> PowerPlan:
        if not self.enabled:
            return PowerPlan(0.0, target_watts, False)
        measured = self.measure(model)
        if measured > target_watts:
            self.bound(model, target_watts)
            return PowerPlan(measured, target_watts, True)
        return PowerPlan(measured, target_watts, False)
