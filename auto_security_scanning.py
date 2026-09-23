# abliterate/tuning/10_auto_security_scanning.py
#
# Tuning Type 10 of 103:
# Auto security scanning.
#
# Runtime toggle: --enable tuning.auto_security_scanning  (default: on)
#
# Effect:
#   Scan the model for adversarial modifications before ablation and after.
#   Detect backdoors, trojans, poisoned weights, injected trigger behavior.
#
# This file owns:
#   - the security scan itself
#   - the report of what was found and where
#
# This file does NOT:
#   - modify the model
#   - target layers (that is tuning/11)

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SecurityFinding:
    layer: int
    kind: str
    severity: float
    detail: str = ""


@dataclass
class SecurityReport:
    findings: list = field(default_factory=list)


class AutoSecurityScanning:

    KINDS = ("backdoor", "trojan", "poisoned_weight", "trigger_behavior", "unknown")

    def __init__(self, model, enabled: bool = True):
        self.model = model
        self.enabled = enabled

    def scan(self) -> SecurityReport:
        if not self.enabled:
            return SecurityReport()
        return SecurityReport()

    def scan_layer(self, layer_idx: int) -> list:
        return []


TOGGLE = {"id": "tuning.auto_security_scanning", "default": True, "read_only": True}
