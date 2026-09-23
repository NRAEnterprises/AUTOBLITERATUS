# abliterate/tuning/83_adversarial_red_team_auto_tuning.py
#
# Tuning Type 83 of 103:
# Adversarial red-team auto tuning.
#
# Runtime toggle: --enable tuning.adversarial_red_team_auto  (default: on)
#
# Effect:
#   Generate adversarial prompts targeting the ablated model and verify
#   behavior.
#
# This file owns:
#   - adversarial prompt generation
#   - response analysis
#   - report

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class RedTeamReport:
    probes: list = field(default_factory=list)
    findings: list = field(default_factory=list)


class AdversarialRedTeamAutoTuning:

    def __init__(self, generate: Callable, analyze: Callable,
                 enabled: bool = True):
        self.generate = generate
        self.analyze = analyze
        self.enabled = enabled

    def run(self, model, tokenizer) -> RedTeamReport:
        if not self.enabled:
            return RedTeamReport()
        probes = self.generate()
        findings = [self.analyze(model, tokenizer, p) for p in probes]
        return RedTeamReport(probes=probes, findings=findings)


TOGGLE = {"id": "tuning.adversarial_red_team_auto", "default": True, "read_only": True}
