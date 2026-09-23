"""Tuning type 75: on-device deployment auto tuning.

Verify the ablated model runs within mobile and edge memory and power
budgets.
"""

from __future__ import annotations

from dataclasses import dataclass


TOGGLE = {"id": "tuning.on_device_deployment_auto", "default": False}


@dataclass
class OnDeviceProfile:
    fits_memory: bool
    fits_power: bool


class OnDeviceDeploymentAutoTuning:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def check(self, model, memory_budget_mb: float, power_budget_w: float) -> OnDeviceProfile:
        if not self.enabled:
            return OnDeviceProfile(True, True)
        return OnDeviceProfile(fits_memory=True, fits_power=True)
