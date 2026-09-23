"""Tuning type 83: adversarial red-team auto tuning.

Generate adversarial prompts targeting the ablated model and verify
behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


TOGGLE = {"id": "tuning.adversarial_red_team_auto", "default": True}


@dataclass
class RedTeamReport:
    probes: list = field(default_factory=list)
    findings: list = field(default_factory=list)


class AdversarialRedTeamAutoTuning:
    def __init__(self, generate: Callable, analyze: Callable, enabled: bool = True) -> None:
        self.generate = generate
        self.analyze = analyze
        self.enabled = enabled

    def run(self, model, tokenizer) -> RedTeamReport:
        if not self.enabled:
            return RedTeamReport()
        probes = self.generate()
        findings = [self.analyze(model, tokenizer, p) for p in probes]
        return RedTeamReport(probes=probes, findings=findings)
