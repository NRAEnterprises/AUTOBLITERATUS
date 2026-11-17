"""Capture a validated causal-LM metric baseline before model transformations."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from autobliteratus.tuning.evaluation_metrics import (
    evaluate_causal_lm,
    load_evaluation_texts,
)

TOGGLE = {
    "id": "tuning.baseline_auto_pre_post",
    "default": False,
    "desc": "Measure explicit evaluation prompts on the untouched model before tuning for a paired regression gate.",
}


@dataclass
class BaselinePair:
    pre: dict[str, float] = field(default_factory=dict)
    post: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, dict[str, Any]] = field(default_factory=dict)


def validate_metrics(metrics: Any, *, label: str) -> dict[str, float]:
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError(f"{label} must be a non-empty metric mapping.")
    validated: dict[str, float] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} metric names must be non-empty strings.")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{label} metric {name!r} must be a finite number.")
        validated[name] = float(value)
    return validated


class BaselineAutoTuningPrePost:
    """Small reusable pair collector; the pipeline stage uses the built-in battery."""

    def __init__(self, battery_measure: Callable | None = None, tolerance_ratio: float = 0.02) -> None:
        if not math.isfinite(tolerance_ratio) or not 0.0 <= tolerance_ratio <= 1.0:
            raise ValueError("tolerance_ratio must be finite and in [0, 1].")
        self.measure = battery_measure
        self.tolerance_ratio = tolerance_ratio

    def _measure(self, model: Any, tokenizer: Any) -> dict[str, float]:
        if self.measure is None:
            raise RuntimeError("A battery_measure callback is required for direct class use.")
        return validate_metrics(self.measure(model, tokenizer), label="measured metrics")

    def capture_pre(self, model: Any, tokenizer: Any) -> dict[str, float]:
        return self._measure(model, tokenizer)

    def capture_post(self, model: Any, tokenizer: Any) -> dict[str, float]:
        return self._measure(model, tokenizer)

    def derive_thresholds(
        self,
        pre: dict[str, float],
        post: dict[str, float],
        tolerance_ratio: float | None = None,
    ) -> dict[str, dict[str, float]]:
        before = validate_metrics(pre, label="pre metrics")
        after = validate_metrics(post, label="post metrics")
        if set(before) != set(after):
            raise ValueError("Pre/post metric names must match exactly.")
        ratio = self.tolerance_ratio if tolerance_ratio is None else tolerance_ratio
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(float(ratio)) or not 0 <= ratio <= 1:
            raise ValueError("tolerance_ratio must be finite and in [0, 1].")
        return {
            name: {
                "baseline": baseline,
                "post": after[name],
                "absolute_tolerance": abs(baseline) * float(ratio),
                "tolerance_ratio": float(ratio),
            }
            for name, baseline in before.items()
        }

    def run(self, model_pre: Any, model_post: Any, tokenizer: Any) -> BaselinePair:
        pre = self.capture_pre(model_pre, tokenizer)
        post = self.capture_post(model_post, tokenizer)
        return BaselinePair(pre=pre, post=post, thresholds=self.derive_thresholds(pre, post))


def run(adapter: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the loaded, pre-edit model and preserve only numeric results."""
    texts = load_evaluation_texts(context)
    max_length = int(context.get("evaluation_max_length", 256))
    evaluation = evaluate_causal_lm(
        adapter.handle.model,
        adapter.handle.tokenizer,
        texts,
        max_length=max_length,
    )
    metrics = validate_metrics(evaluation["metrics"], label="baseline metrics")
    report = {
        "status": "baseline_recorded",
        "metrics": metrics,
        "sample_count": evaluation["sample_count"],
        "scored_tokens": evaluation["scored_tokens"],
        "evaluation_max_length": max_length,
        "perplexity_cap_nll": evaluation["perplexity_cap_nll"],
        "model_modified": False,
    }
    context["baseline_metrics"] = metrics
    context["baseline_report"] = report
    return report
