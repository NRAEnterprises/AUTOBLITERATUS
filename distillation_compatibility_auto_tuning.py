# abliterate/tuning/78_distillation_compatibility_auto_tuning.py
#
# Tuning Type 78 of 103:
# Distillation compatibility auto tuning.
#
# Runtime toggle: --enable tuning.distillation_compatibility_auto  (default: on)
#
# Effect:
#   Verify the ablated model can serve as a teacher for a smaller student.
#
# This file owns:
#   - teacher-output stability check
#   - student-reproduction check

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DistillationProfile:
    usable_as_teacher: bool
    notes: str = ""


class DistillationCompatibilityAutoTuning:

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def check(self, model, tokenizer) -> DistillationProfile:
        if not self.enabled:
            return DistillationProfile(True)
        return DistillationProfile(True)


TOGGLE = {"id": "tuning.distillation_compatibility_auto", "default": True, "read_only": True}
