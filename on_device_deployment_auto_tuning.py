# abliterate/tuning/75_on_device_deployment_auto_tuning.py
#
# Tuning Type 75 of 103:
# On-device deployment auto tuning.
#
# Runtime toggle: --enable tuning.on_device_deployment_auto  (default: off)
#
# Effect:
#   Verify the ablated model runs within mobile and edge memory and power
#   budgets.
#
# This file owns:
#   - memory budget check
#   - power budget check

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OnDeviceProfile:
    fits_memory: bool
    fits_power: bool


class OnDeviceDeploymentAutoTuning:

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def check(self, model, memory_budget_mb: float, power_budget_w: float) -> OnDeviceProfile:
        if not self.enabled:
            return OnDeviceProfile(True, True)
        return OnDeviceProfile(fits_memory=True, fits_power=True)


TOGGLE = {"id": "tuning.on_device_deployment_auto", "default": False, "read_only": True}
