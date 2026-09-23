# abliterate/tuning/08_adversarial_auto_tuning.py
#
# Tuning Type 8 of 103:
# Adversarial auto tuning.
#
# Runtime toggle: --enable tuning.adversarial_auto  (default: on)
#
# Effect:
#   Escalate to a more aggressive ablation method when refusal persists after
#   the standard pass. De-escalate when refusal is removed but capability is
#   still intact. The choice of method escalation is automatic, driven by
#   the refusal persistence measurement.
#
# This file owns:
#   - the persistence measurement
#   - the escalation ladder
#   - the de-escalation rule
#
# This file consumes:
#   - the refusal probe (from PROBE, passed in)
#   - the ablation methods (from the method registry, passed in)

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class EscalationResult:
    method_chosen: str
    persistence_measured: float
    escalated: bool
    de_escalated: bool


class AdversarialAutoTuning:

    LADDER = ("basic", "advanced", "surgical", "aggressive", "nuclear")

    def __init__(self, refusal_probe: Callable, capability_measure: Callable,
                 enabled: bool = True, persistence_threshold: float = 0.10):
        self.refusal_probe = refusal_probe
        self.capability_measure = capability_measure
        self.enabled = enabled
        self.persistence_threshold = persistence_threshold

    def choose_method(self, model, tokenizer, current_method: str) -> EscalationResult:
        if not self.enabled:
            return EscalationResult(current_method, 0.0, False, False)

        persistence = self.refusal_probe(model, tokenizer)
        capability = self.capability_measure(model, tokenizer)

        if persistence > self.persistence_threshold:
            return EscalationResult(self._step_up(current_method), persistence, True, False)

        if persistence <= self.persistence_threshold and capability.get("coherence", 0.0) > 0.95:
            return EscalationResult(self._step_down(current_method), persistence, False, True)

        return EscalationResult(current_method, persistence, False, False)

    def _step_up(self, method: str) -> str:
        idx = self.LADDER.index(method) if method in self.LADDER else 0
        return self.LADDER[min(idx + 1, len(self.LADDER) - 1)]

    def _step_down(self, method: str) -> str:
        idx = self.LADDER.index(method) if method in self.LADDER else 0
        return self.LADDER[max(idx - 1, 0)]


TOGGLE = {"id": "tuning.adversarial_auto", "default": True}
