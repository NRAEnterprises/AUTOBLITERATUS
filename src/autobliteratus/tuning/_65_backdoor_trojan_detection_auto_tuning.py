"""Tuning type 65: backdoor and trojan detection auto tuning.

Scan for injected behavior that activates on trigger phrases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


TOGGLE = {"id": "tuning.backdoor_trojan_detection", "default": True}


@dataclass
class BackdoorReport:
    detected: bool
    triggers: list = field(default_factory=list)
    layers: list = field(default_factory=list)


class BackdoorTrojanDetectionAutoTuning:
    def __init__(self, probe: Callable, enabled: bool = True) -> None:
        self.probe = probe
        self.enabled = enabled

    def run(self, model, tokenizer) -> BackdoorReport:
        if not self.enabled:
            return BackdoorReport(False)
        triggers, layers = self.probe(model, tokenizer)
        return BackdoorReport(detected=bool(triggers), triggers=triggers, layers=layers)
