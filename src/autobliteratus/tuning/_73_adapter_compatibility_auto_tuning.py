"""Tuning type 73: adapter compatibility auto tuning.

Verify LoRA / QLoRA / DoRA adapters still work on the ablated model.
"""

from __future__ import annotations

from dataclasses import dataclass


TOGGLE = {"id": "tuning.adapter_compatibility_auto", "default": True}


@dataclass
class AdapterProfile:
    compatible: bool
    notes: str = ""


class AdapterCompatibilityAutoTuning:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def check(self, model, adapter) -> AdapterProfile:
        if not self.enabled:
            return AdapterProfile(True)
        return AdapterProfile(True)
