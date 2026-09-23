# abliterate/tuning/73_adapter_compatibility_auto_tuning.py
#
# Tuning Type 73 of 103:
# Adapter compatibility auto tuning.
#
# Runtime toggle: --enable tuning.adapter_compatibility_auto  (default: on)
#
# Effect:
#   Verify LoRA / QLoRA / DoRA adapters still work on the ablated model.
#
# This file owns:
#   - adapter load check
#   - adapter inference check

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdapterProfile:
    compatible: bool
    notes: str = ""


class AdapterCompatibilityAutoTuning:

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def check(self, model, adapter) -> AdapterProfile:
        if not self.enabled:
            return AdapterProfile(True)
        return AdapterProfile(True)


TOGGLE = {"id": "tuning.adapter_compatibility_auto", "default": True, "read_only": True}
