"""Tuning type 9: multi objective auto tuning.

Pareto frontier over multiple objectives at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


TOGGLE = {"id": "tuning.multi_objective_auto", "default": True}


@dataclass
class ParetoPoint:
    config: dict
    scores: dict


class MultiObjectiveAutoTuning:
    def __init__(self, evaluate: Callable, objectives: Iterable[str], enabled: bool = True) -> None:
        self.evaluate = evaluate
        self.objectives = tuple(objectives)
        self.enabled = enabled

    def search(self, candidates: Iterable[dict]) -> list[ParetoPoint]:
        if not self.enabled:
            return []
        points = [ParetoPoint(config=c, scores=self.evaluate(c)) for c in candidates]
        return self._pareto_front(points)

    def choose(self, frontier: list[ParetoPoint]) -> ParetoPoint:
        if not frontier:
            raise ValueError("Empty frontier")
        return frontier[0]

    def _pareto_front(self, points: list[ParetoPoint]) -> list[ParetoPoint]:
        front = []
        for p in points:
            if not any(self._dominates(q.scores, p.scores) for q in points if q is not p):
                front.append(p)
        return front

    def _dominates(self, a: dict, b: dict) -> bool:
        better_or_equal = all(a.get(o, 0.0) >= b.get(o, 0.0) for o in self.objectives)
        strictly_better = any(a.get(o, 0.0) > b.get(o, 0.0) for o in self.objectives)
        return better_or_equal and strictly_better
