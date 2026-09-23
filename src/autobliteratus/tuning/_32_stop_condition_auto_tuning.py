"""Tuning type 32: stop condition auto tuning.

Plateau detection and ceiling detection for the correction loop.
"""

from __future__ import annotations

from dataclasses import dataclass


TOGGLE = {"id": "tuning.stop_condition_auto", "default": True}


@dataclass
class StopDecision:
    stop: bool
    reason: str


class StopConditionAutoTuning:
    def __init__(
        self,
        enabled: bool = True,
        plateau_window: int = 3,
        plateau_epsilon: float = 0.001,
    ) -> None:
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
        if all(v <= 0 for v in current_deficit.values()):
            return StopDecision(True, "all_axes_restored")
        return StopDecision(False, "progressing")
