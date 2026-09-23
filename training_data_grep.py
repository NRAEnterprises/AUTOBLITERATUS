# abliterate/tuning/05_training_data_grep.py
#
# Tuning Type 5 of 103:
# Training data greped.
#
# Runtime toggle: --enable tuning.training_data_grep  (default: on)
#
# Effect:
#   Search the model's weights for training-data signatures. Read-only.
#   Query in, ranked training-data matches out. No modification to the model.
#
# This file owns:
#   - the query interface
#   - gradient-direction matching against the model's weight delta
#   - composition read (WARP-style) of what the model was trained on
#
# This file consumes:
#   - the base model (pre-fine-tune reference), if available
#   - the model under inspection

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


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

    def __init__(self, model, base_model=None, batch_size: int = 32):
        self.model = model
        self.base_model = base_model
        self.batch_size = batch_size

    def grep(self, candidates: Iterable[str], top_k: int = 50) -> list:
        scored = []
        for sample in candidates:
            score = self._gradient_direction_score(sample)
            scored.append((sample, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [GrepMatch(sample=s, score=score) for s, score in scored[:top_k]]

    def composition(self) -> CompositionReport:
        if self.base_model is None:
            return CompositionReport()
        return CompositionReport()

    def _gradient_direction_score(self, sample: str) -> float:
        return 0.0


TOGGLE = {"id": "tuning.training_data_grep", "default": True, "read_only": True}
