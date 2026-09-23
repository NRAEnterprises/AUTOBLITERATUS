# abliterate/tuning/22_m2m_optimizational_auto_tuning.py
#
# Tuning Type 22 of 103:
# M2M optimizational auto tuning.
#
# Runtime toggle: --enable tuning.m2m_optimizational_auto  (default: on)
#
# Effect:
#   Model-to-model optimization loop. A separate model evaluates the ablated
#   model's outputs and feeds corrections back into the tuning loop.
#
# This file owns:
#   - the evaluator model call
#   - the feedback extraction
#   - the routing of feedback into the correction interface

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class M2MFeedback:
    axis: str
    deficit: float
    note: str = ""


class M2MOptimizationalAutoTuning:

    def __init__(self, evaluator: Callable, enabled: bool = True):
        self.evaluator = evaluator
        self.enabled = enabled

    def evaluate(self, model, tokenizer, prompt_set: list) -> list:
        if not self.enabled:
            return []
        feedback = []
        for prompt in prompt_set:
            result = self.evaluator(model=model, tokenizer=tokenizer, prompt=prompt)
            for axis, deficit in result.items():
                feedback.append(M2MFeedback(axis=axis, deficit=float(deficit)))
        return feedback


TOGGLE = {"id": "tuning.m2m_optimizational_auto", "default": True}
