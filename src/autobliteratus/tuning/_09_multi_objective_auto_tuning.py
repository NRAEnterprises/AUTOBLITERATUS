"""Deterministic Pareto analysis for explicitly supplied tuning candidates.

The stage evaluates no model changes by itself: candidate generation/evaluation
must be performed by a caller and supplied as finite objective scores. This
module validates those scores, computes a Pareto frontier, and selects a
weighted normalized compromise without pretending to run an optimizer.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

TOGGLE = {
    "id": "tuning.multi_objective_auto",
    "default": False,
    "desc": "Validate candidate objective scores, compute a Pareto frontier, and select a deterministic compromise.",
}


@dataclass(frozen=True)
class ParetoPoint:
    config: dict[str, Any]
    scores: dict[str, float]
    candidate_id: str


class MultiObjectiveAutoTuning:
    def __init__(
        self,
        objectives: dict[str, str],
        weights: dict[str, float] | None = None,
        enabled: bool = True,
    ) -> None:
        if not objectives or any(direction not in {"max", "min"} for direction in objectives.values()):
            raise ValueError("objectives must map at least one name to 'max' or 'min'.")
        self.objectives = dict(objectives)
        self.weights = {name: float((weights or {}).get(name, 1.0)) for name in objectives}
        if any(not math.isfinite(weight) or weight < 0 for weight in self.weights.values()):
            raise ValueError("Objective weights must be finite and non-negative.")
        if not any(weight > 0 for weight in self.weights.values()):
            raise ValueError("At least one objective weight must be positive.")
        self.enabled = enabled

    def search(self, candidates: Iterable[dict[str, Any]]) -> list[ParetoPoint]:
        if not self.enabled:
            return []
        points: list[ParetoPoint] = []
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise TypeError(f"Candidate {index} must be a mapping.")
            scores = candidate.get("scores")
            config = candidate.get("config", {})
            if not isinstance(config, dict) or not isinstance(scores, dict):
                raise TypeError(f"Candidate {index} must contain mapping-valued 'config' and 'scores'.")
            missing = sorted(set(self.objectives) - set(scores))
            if missing:
                raise ValueError(f"Candidate {index} is missing objective scores: {missing}.")
            normalized_scores: dict[str, float] = {}
            for name in self.objectives:
                value = scores[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(f"Candidate {index} objective {name!r} must be a finite number.")
                normalized_scores[name] = float(value)
            points.append(
                ParetoPoint(
                    config=dict(config),
                    scores=normalized_scores,
                    candidate_id=str(candidate.get("id", index)),
                )
            )
        return self._pareto_front(points)

    def choose(self, frontier: list[ParetoPoint]) -> ParetoPoint:
        if not frontier:
            raise ValueError("Cannot choose from an empty Pareto frontier.")
        bounds = {
            name: (
                min(point.scores[name] for point in frontier),
                max(point.scores[name] for point in frontier),
            )
            for name in self.objectives
        }

        def utility(point: ParetoPoint) -> float:
            weighted = 0.0
            total_weight = 0.0
            for name, direction in self.objectives.items():
                weight = self.weights[name]
                if weight == 0:
                    continue
                low, high = bounds[name]
                score = point.scores[name]
                normalized = 1.0 if high == low else (score - low) / (high - low)
                if direction == "min":
                    normalized = 1.0 - normalized
                weighted += weight * normalized
                total_weight += weight
            return weighted / total_weight

        return max(frontier, key=utility)

    def _dominates(self, a: dict[str, float], b: dict[str, float]) -> bool:
        better_or_equal = True
        strictly_better = False
        for name, direction in self.objectives.items():
            if direction == "max":
                better_or_equal &= a[name] >= b[name]
                strictly_better |= a[name] > b[name]
            else:
                better_or_equal &= a[name] <= b[name]
                strictly_better |= a[name] < b[name]
        return better_or_equal and strictly_better

    def _pareto_front(self, points: list[ParetoPoint]) -> list[ParetoPoint]:
        return [
            point
            for index, point in enumerate(points)
            if not any(
                self._dominates(other.scores, point.scores)
                for other_index, other in enumerate(points)
                if index != other_index
            )
        ]


def run(adapter: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Select from pre-evaluated candidates; never mutates the loaded model."""
    candidates = context.get("multi_objective_candidates", [])
    if candidates is None:
        candidates = []
    if not isinstance(candidates, (list, tuple)):
        raise TypeError("multi_objective_candidates must be a list of candidate mappings.")
    if not candidates:
        report = {
            "status": "no_candidates",
            "candidate_count": 0,
            "pareto_frontier": [],
            "selected_candidate": None,
            "model_modified": False,
        }
        context["multi_objective_report"] = report
        return report

    objectives = context.get("multi_objective_objectives", {"quality": "max"})
    weights = context.get("multi_objective_weights")
    if not isinstance(objectives, dict):
        raise TypeError("multi_objective_objectives must map objective names to 'max' or 'min'.")
    optimizer = MultiObjectiveAutoTuning(objectives=objectives, weights=weights)
    frontier = optimizer.search(candidates)
    selected = optimizer.choose(frontier)
    report = {
        "status": "selected",
        "candidate_count": len(candidates),
        "pareto_frontier": [
            {"candidate_id": point.candidate_id, "config": point.config, "scores": point.scores}
            for point in frontier
        ],
        "selected_candidate": {
            "candidate_id": selected.candidate_id,
            "config": selected.config,
            "scores": selected.scores,
        },
        "objectives": objectives,
        "weights": optimizer.weights,
        "model_modified": False,
        "interpretation": "Selection among supplied scored candidates only; no model search or evaluation was run.",
    }
    context["multi_objective_report"] = report
    return report
