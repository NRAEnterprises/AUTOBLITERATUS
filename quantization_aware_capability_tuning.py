# abliterate/tuning/29_quantization_aware_capability_tuning.py
#
# Tuning Type 29 of 103:
# Quantization aware capability tuning.
#
# Runtime toggle: --enable tuning.quantization_aware_capability  (default: on)
#
# Effect:
#   Same as tuning/03 but explicitly coordinated with the scheme selected by
#   tuning/28. Where 03 applies the scheme, 29 verifies that the tuning pass
#   itself was executed at the scheme's precision end to end.
#
# This file owns:
#   - the precision end-to-end guarantee
#   - the verification that no step in the tuning ran in a different precision
#
# This file consumes:
#   - the scheme selection (from tuning/28, passed in)
#   - the tuning pass (from tuning/03, passed in)

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PrecisionVerification:
    all_steps_at_target_precision: bool
    violations: list


class QuantizationAwareCapabilityTuning:

    def __init__(self, scheme_selection, enabled: bool = True):
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


TOGGLE = {"id": "tuning.quantization_aware_capability", "default": True}
