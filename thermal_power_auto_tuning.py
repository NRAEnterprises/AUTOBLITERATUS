# abliterate/tuning/46_thermal_power_auto_tuning.py
#
# Tuning Type 46 of 103:
# Thermal / power auto tuning.
#
# Runtime toggle: --enable tuning.thermal_power_auto  (default: off)
#
# Effect:
#   For edge and mobile deployment, power draw measured and bounded.
#
# This file owns:
#   - power draw measurement
#   - power bounding

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PowerPlan:
    watts_measured: float
    watts_bound: float
    bounded: bool


class ThermalPowerAutoTuning:

    def __init__(self, measure, bound, enabled: bool = False):
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


TOGGLE = {"id": "tuning.thermal_power_auto", "default": False}
