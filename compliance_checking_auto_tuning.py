# abliterate/tuning/90_compliance_checking_auto_tuning.py
#
# Tuning Type 90 of 103:
# Compliance-checking auto tuning.
#
# Runtime toggle: --enable tuning.compliance_checking_auto  (default: on)
#
# Effect:
#   Verify the ablated model meets user-defined compliance rules.
#
# This file owns:
#   - compliance rule loading
#   - rule evaluation
#   - report

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ComplianceReport:
    rules_passed: list = field(default_factory=list)
    rules_failed: list = field(default_factory=list)


class ComplianceCheckingAutoTuning:

    def __init__(self, rules: dict, evaluate: Callable, enabled: bool = True):
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


TOGGLE = {"id": "tuning.compliance_checking_auto", "default": True}
