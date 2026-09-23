# abliterate/tuning/53_ouroboros_loop_escalation_auto_tuning.py
#
# Tuning Type 53 of 103:
# Ouroboros loop escalation auto tuning.
#
# Runtime toggle: --enable tuning.ouroboros_loop_escalation  (default: on)
#
# Effect:
#   When guardrails self-repair, escalate correction strength automatically.
#
# This file owns:
#   - self-repair detection
#   - escalation of correction strength

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class OuroborosEscalation:
    self_repair_detected: bool
    escalated: bool
    new_strength: float


class OuroborosLoopEscalationAutoTuning:

    def __init__(self, detect_self_repair: Callable, enabled: bool = True):
        self.detect_self_repair = detect_self_repair
        self.enabled = enabled

    def run(self, model, current_strength: float) -> OuroborosEscalation:
        if not self.enabled:
            return OuroborosEscalation(False, False, current_strength)
        repairing = self.detect_self_repair(model)
        if repairing:
            return OuroborosEscalation(True, True, min(2.0, current_strength * 1.25))
        return OuroborosEscalation(False, False, current_strength)


TOGGLE = {"id": "tuning.ouroboros_loop_escalation", "default": True}
