"""Tuning type 90: compliance-checking auto tuning.

Verify the ablated model meets user-defined compliance rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


TOGGLE = {"id": "tuning.compliance_checking_auto", "default": True}


@dataclass
class ComplianceReport:
    rules_passed: list = field(default_factory=list)
    rules_failed: list = field(default_factory=list)


class ComplianceCheckingAutoTuning:
    def __init__(self, rules: dict, evaluate: Callable, enabled: bool = True) -> None:
        self.rules = rules
        self.evaluate = evaluate
        self.enabled = enabled

    def run(self, model, tokenizer) -> ComplianceReport:
        if not self.enabled:
            return ComplianceReport()
        passed, failed = [], []
        for name, rule in self.rules.items():
            (passed if self.evaluate(model, tokenizer, rule) else failed).append(name)
        return ComplianceReport(rules_passed=passed, rules_failed=failed)
