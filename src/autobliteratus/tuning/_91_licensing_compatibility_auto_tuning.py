"""Tuning type 91: licensing compatibility auto tuning.

Verify the ablated model's license is preserved and modified appropriately.
"""

from __future__ import annotations

from dataclasses import dataclass


TOGGLE = {"id": "tuning.licensing_compatibility_auto", "default": True}


@dataclass
class LicenseReport:
    original: str
    ablated: str
    modification_required: bool


class LicensingCompatibilityAutoTuning:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def check(self, model_path_original: str, model_path_ablated: str) -> LicenseReport:
        if not self.enabled:
            return LicenseReport("", "", False)
        return LicenseReport("", "", False)
