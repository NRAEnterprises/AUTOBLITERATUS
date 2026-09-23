"""Tuning type 22: M2M optimizational auto tuning.

Model-to-model optimization loop. A separate model evaluates the ablated
model's outputs and feeds corrections back into the tuning loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


TOGGLE = {"id": "tuning.m2m_optimizational_auto", "default": True}


@dataclass
class M2MFeedback:
    axis: str
    deficit: float
    note: str = ""


class M2MOptimizationalAutoTuning:
    def __init__(self, evaluator: Callable, enabled: bool = True) -> None:
        self.evaluator = evaluator
        self.enabled = enabled

    def evaluate(self, model, tokenizer, prompt_set: list) -> list[M2MFeedback]:
        if not self.enabled:
            return []
        feedback: list[M2MFeedback] = []
        for prompt in prompt_set:
            result = self.evaluator(model=model, tokenizer=tokenizer, prompt=prompt)
            for axis, deficit in result.items():
                feedback.append(M2MFeedback(axis=axis, deficit=float(deficit)))
        return feedback
