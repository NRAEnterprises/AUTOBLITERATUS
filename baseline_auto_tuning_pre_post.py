# abliterate/tuning/07_baseline_auto_tuning_pre_post.py
#
# Tuning Type 7 of 103:
# Baseline auto tuning pre and post ablation.
#
# Runtime toggle: --enable tuning.baseline_auto_pre_post  (default: on)
#
# Effect:
#   Capture the model's capability battery scores before ablation and again
#   after. Store both. Every downstream threshold is derived from these
#   numbers, not from a global constant or a per-family default.
#
# This file owns:
#   - the pre-ablation capture
#   - the post-ablation capture
#   - the storage of both
#   - the derivation of thresholds from the two
#
# This file consumes:
#   - the capability battery (from tuning/06, passed in)

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class BaselinePair:
    pre: dict = field(default_factory=dict)
    post: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)


class BaselineAutoTuningPrePost:

    def __init__(self, battery_measure: Callable, work_dir: str = "./abliterate_workspace"):
        self.measure = battery_measure
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def capture_pre(self, model, tokenizer) -> dict:
        scores = self.measure(model, tokenizer)
        return scores

    def capture_post(self, model, tokenizer) -> dict:
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


TOGGLE = {"id": "tuning.baseline_auto_pre_post", "default": True}
