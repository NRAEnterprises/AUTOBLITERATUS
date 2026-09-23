# abliterate/tuning/11_auto_security_tuning.py
#
# Tuning Type 11 of 103:
# Auto security tuning.
#
# Runtime toggle: --enable tuning.auto_security_tuning  (default: on)
#
# Effect:
#   Target the specific layers flagged by the security scan.
#   Apply corrections to those layers only.
#
# This file owns:
#   - mapping findings to layers
#   - applying the correction to those layers
#
# This file consumes:
#   - the security report (from tuning/10, passed in)
#   - a correction callable (from correction-primitive files)

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class SecurityTuningResult:
    layers_targeted: list
    corrections_applied: int


class AutoSecurityTuning:

    def __init__(self, apply_correction: Callable, enabled: bool = True):
        self.apply_correction = apply_correction
        self.enabled = enabled

    def run(self, model, security_report) -> SecurityTuningResult:
        if not self.enabled:
            return SecurityTuningResult([], 0)
        targeted = sorted({f.layer for f in security_report.findings})
        for layer_idx in targeted:
            for finding in [f for f in security_report.findings if f.layer == layer_idx]:
                self.apply_correction(
                    model=model,
                    layer_idx=layer_idx,
                    kind=finding.kind,
                    severity=finding.severity,
                )
        return SecurityTuningResult(targeted, len(targeted))


TOGGLE = {"id": "tuning.auto_security_tuning", "default": True}
