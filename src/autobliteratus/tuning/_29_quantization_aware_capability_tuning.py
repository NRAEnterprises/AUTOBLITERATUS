"""Tuning type 29: quantization aware capability tuning.

Verify the tuning pass itself ran end to end at the target precision.
"""

from __future__ import annotations

from dataclasses import dataclass


TOGGLE = {"id": "tuning.quantization_aware_capability", "default": True}


@dataclass
class PrecisionVerification:
    all_steps_at_target_precision: bool
    violations: list


class QuantizationAwareCapabilityTuning:
    def __init__(self, scheme_selection, enabled: bool = True) -> None:
        self.scheme_selection = scheme_selection
        self.enabled = enabled

    def verify(self, step_log: list) -> PrecisionVerification:
        if not self.enabled:
            return PrecisionVerification(True, [])
        violations = []
        for step in step_log:
            expected = self.scheme_selection.per_layer.get(step.get("layer"), {}).get("scheme", "none")
            actual = step.get("scheme", "none")
            if expected != actual:
                violations.append({
                    "layer": step.get("layer"),
                    "expected": expected,
                    "actual": actual,
                    "step": step.get("name"),
                })
        return PrecisionVerification(
            all_steps_at_target_precision=not violations,
            violations=violations,
        )
