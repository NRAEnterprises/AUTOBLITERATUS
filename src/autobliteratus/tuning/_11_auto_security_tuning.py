"""Tuning type 11: auto security tuning.

Target the specific layers flagged by the security scan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


TOGGLE = {"id": "tuning.auto_security_tuning", "default": True}


@dataclass
class SecurityTuningResult:
    layers_targeted: list[int]
    corrections_applied: int


class AutoSecurityTuning:
    def __init__(self, apply_correction: Callable, enabled: bool = True) -> None:
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
