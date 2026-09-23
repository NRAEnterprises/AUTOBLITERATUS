# abliterate/tuning/09_multi_objective_auto_tuning.py
#
# Tuning Type 9 of 103:
# Multi objective auto tuning.
#
# Runtime toggle: --enable tuning.multi_objective_auto  (default: on)
#
# Effect:
#   Optimize under multiple objectives at once. Refusal removal, coherence,
#   reasoning, calibration, and every other measured axis. Not a single
#   scalar. A Pareto frontier over the objectives.
#
# This file owns:
#   - the objective list
#   - the Pareto construction
#   - the choice of the Pareto point to apply
#
# This file consumes:
#   - the capability battery (from tuning/06)
#   - the refusal probe (from PROBE)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable


@dataclass
class ParetoPoint:
    config: dict
    scores: dict


class MultiObjectiveAutoTuning:

    def __init__(self, evaluate: Callable, objectives: Iterable[str],
                 enabled: bool = True):
        self.evaluate = evaluate
        self.objectives = tuple(objectives)
        self.enabled = enabled

    def search(self, candidates: Iterable[dict]) -> list:
        if not self.enabled:
            return []
        points = []
        for candidate in candidates:
            scores = self.evaluate(candidate)
            points.append(ParetoPoint(config=candidate, scores=scores))
        return self._pareto_front(points)

    def choose(self, frontier: list) -> ParetoPoint:
        if not frontier:
            raise ValueError("Empty frontier")
        return frontier[0]

    def _pareto_front(self, points: list) -> list:
        front = []
        for p in points:
            dominated = False
            for q in points:
                if q is p:
                    continue
                if self._dominates(q.scores, p.scores):
                    dominated = True
                    break
            if not dominated:
                front.append(p)
        return front

    def _dominates(self, a: dict, b: dict) -> bool:
        better_or_equal = all(a.get(o, 0.0) >= b.get(o, 0.0) for o in self.objectives)
        strictly_better = any(a.get(o, 0.0) > b.get(o, 0.0) for o in self.objectives)
        return better_or_equal and strictly_better


TOGGLE = {"id": "tuning.multi_objective_auto", "default": True}
