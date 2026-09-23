"""Tuning type 5: training data greped.

Read-only search of the model's weights for training-data signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


TOGGLE = {"id": "tuning.training_data_grep", "default": True}


@dataclass
class GrepMatch:
    sample: str
    score: float
    domain: str = ""
    layer_hits: list = field(default_factory=list)


@dataclass
class CompositionReport:
    domains: dict = field(default_factory=dict)
    proportions: dict = field(default_factory=dict)


class TrainingDataGrep:
    def __init__(self, model, base_model=None, batch_size: int = 32) -> None:
        self.model = model
        self.base_model = base_model
        self.batch_size = batch_size

    def grep(self, candidates: Iterable[str], top_k: int = 50) -> list[GrepMatch]:
        scored = [(sample, self._gradient_direction_score(sample)) for sample in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [GrepMatch(sample=s, score=score) for s, score in scored[:top_k]]

    def composition(self) -> CompositionReport:
        if self.base_model is None:
            return CompositionReport()
        return CompositionReport()

    def _gradient_direction_score(self, sample: str) -> float:
        return 0.0
