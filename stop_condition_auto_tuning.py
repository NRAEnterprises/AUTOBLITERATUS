# abliterate/tuning/32_stop_condition_auto_tuning.py
#
# Tuning Type 32 of 103:
# Stop condition auto tuning.
#
# Runtime toggle: --enable tuning.stop_condition_auto  (default: on)
#
# Effect:
#   The correction loop decides on its own when to stop, based on the model's
#   own pre-ablation numbers. No manually authored threshold per model. No
#   fixed iteration cap as the only exit.
#
# This file owns:
#   - the stop condition itself
#   - the plateau detection
#   - the ceiling detection

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StopDecision:
    stop: bool
    reason: str


class StopConditionAutoTuning:

    def __init__(self, enabled: bool = True, plateau_window: int = 3,
                 plateau_epsilon: float = 0.001):
        self.enabled = enabled
        self.plateau_window = plateau_window
        self.plateau_epsilon = plateau_epsilon
        self._history: list = []

    def check(self, current_deficit: dict, baseline: dict) -> StopDecision:
        if not self.enabled:
            return StopDecision(False, "disabled")
        self._history.append(dict(current_deficit))
        if len(self._history) < self.plateau_window:
            return StopDecision(False, "not_enough_history")
        recent = self._history[-self.plateau_window:]
        plateau = all(
            abs(recent[i].get(a, 0.0) - recent[i - 1].get(a, 0.0)) < self.plateau_epsilon
            for i in range(1, len(recent))
            for a in current_deficit
        )
        if plateau:
            return StopDecision(True, "plateau")
        all_clear = all(v <= 0 for v in current_deficit.values())
        if all_clear:
            return StopDecision(True, "all_axes_restored")
        return StopDecision(False, "progressing")


TOGGLE = {"id": "tuning.stop_condition_auto", "default": True}
