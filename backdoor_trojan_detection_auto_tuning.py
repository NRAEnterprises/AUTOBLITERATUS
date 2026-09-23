# abliterate/tuning/65_backdoor_trojan_detection_auto_tuning.py
#
# Tuning Type 65 of 103:
# Backdoor / trojan detection auto tuning.
#
# Runtime toggle: --enable tuning.backdoor_trojan_detection  (default: on)
#
# Effect:
#   Scan for injected behavior that activates on trigger phrases.
#
# This file owns:
#   - trigger phrase probing
#   - activation pattern analysis
#   - report of any detected backdoor

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BackdoorReport:
    detected: bool
    triggers: list = field(default_factory=list)
    layers: list = field(default_factory=list)


class BackdoorTrojanDetectionAutoTuning:

    def __init__(self, probe: Callable, enabled: bool = True):
        self.probe = probe
        self.enabled = enabled

    def run(self, model, tokenizer) -> BackdoorReport:
        if not self.enabled:
            return BackdoorReport(False)
        triggers, layers = self.probe(model, tokenizer)
        return BackdoorReport(detected=bool(triggers), triggers=triggers, layers=layers)


TOGGLE = {"id": "tuning.backdoor_trojan_detection", "default": True, "read_only": True}
