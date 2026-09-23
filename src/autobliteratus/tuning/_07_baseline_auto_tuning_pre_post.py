"""Tuning type 7: baseline auto tuning pre and post ablation.

Capture baseline before ablation and again after. Derive thresholds from
the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


TOGGLE = {"id": "tuning.baseline_auto_pre_post", "default": True}


@dataclass
class BaselinePair:
    pre: dict[str, float] = field(default_factory=dict)
    post: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, dict] = field(default_factory=dict)


class BaselineAutoTuningPrePost:
    def __init__(self, battery_measure: Callable, work_dir: str = "./abliterate_workspace") -> None:
        self.measure = battery_measure
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def capture_pre(self, model, tokenizer) -> dict[str, float]:
        return self.measure(model, tokenizer)

    def capture_post(self, model, tokenizer) -> dict[str, float]:
        return self.measure(model, tokenizer)

    def derive_thresholds(self, pre: dict, post: dict, tolerance_ratio: float = 0.02) -> dict:
        out = {}
        for axis, base in pre.items():
            out[axis] = {
                "baseline": base,
                "tolerance": base * tolerance_ratio,
                "post": post.get(axis, 0.0),
            }
        return out

    def run(self, model_pre, model_post, tokenizer) -> BaselinePair:
        pre = self.capture_pre(model_pre, tokenizer)
        post = self.capture_post(model_post, tokenizer)
        thresholds = self.derive_thresholds(pre, post)
        return BaselinePair(pre=pre, post=post, thresholds=thresholds)
