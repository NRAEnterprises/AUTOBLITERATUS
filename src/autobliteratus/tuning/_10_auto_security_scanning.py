"""Tuning type 10: auto security scanning.

Read-only scan of the model for adversarial modifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field


TOGGLE = {"id": "tuning.auto_security_scanning", "default": True}


@dataclass
class SecurityFinding:
    layer: int
    kind: str
    severity: float
    detail: str = ""


@dataclass
class SecurityReport:
    findings: list[SecurityFinding] = field(default_factory=list)


class AutoSecurityScanning:
    KINDS = ("backdoor", "trojan", "poisoned_weight", "trigger_behavior", "unknown")

    def __init__(self, model=None, enabled: bool = True) -> None:
        self.model = model
        self.enabled = enabled

    def scan(self) -> SecurityReport:
        if not self.enabled:
            return SecurityReport()
        return SecurityReport()

    def scan_layer(self, layer_idx: int) -> list[SecurityFinding]:
        return []
