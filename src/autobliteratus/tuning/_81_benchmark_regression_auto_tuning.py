"""Compare post-edit evaluation metrics to a captured pre-edit baseline."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from autobliteratus.tuning._07_baseline_auto_tuning_pre_post import validate_metrics
from autobliteratus.tuning.evaluation_metrics import (
    evaluate_causal_lm,
    load_evaluation_texts,
)

TOGGLE = {
    "id": "tuning.benchmark_regression_auto",
    "default": False,
    "desc": "Measure post-edit quality/forward throughput against the baseline and fail the run on material regressions.",
}

DEFAULT_DIRECTIONS = {
    "mean_nll": "min",
    "perplexity": "min",
    "forward_tokens_per_second": "max",
}
DEFAULT_TOLERANCES = {
    "mean_nll": 0.02,
    "perplexity": 0.02,
    # Single-run CPU/GPU timing is noisier than token-weighted loss.
    "forward_tokens_per_second": 0.10,
}


class RegressionDetectedError(RuntimeError):
    """Raised when a valid paired metric exceeds its declared regression tolerance."""


@dataclass
class BenchmarkReport:
    per_benchmark: dict[str, dict[str, Any]] = field(default_factory=dict)
    regressions: list[str] = field(default_factory=list)
    status: str = "not_evaluated"


class BenchmarkRegressionAutoTuning:
    """Direction-aware paired comparison with a relative tolerance per metric."""

    def __init__(
        self,
        directions: dict[str, str] | None = None,
        regression_threshold: float = 0.02,
        regression_tolerances: dict[str, float] | None = None,
    ) -> None:
        if isinstance(regression_threshold, bool) or not isinstance(regression_threshold, (int, float)) or not math.isfinite(float(regression_threshold)) or not 0.0 <= regression_threshold <= 1.0:
            raise ValueError("regression_threshold must be finite and in [0, 1].")
        if directions is not None and (not isinstance(directions, dict) or not directions):
            raise ValueError("directions must be a non-empty mapping of metric names to 'min' or 'max'.")
        self.directions = dict(DEFAULT_DIRECTIONS if directions is None else directions)
        if not self.directions or any(
            not isinstance(name, str) or direction not in {"min", "max"}
            for name, direction in self.directions.items()
        ):
            raise ValueError("directions must map metric names to 'min' or 'max'.")
        self.regression_threshold = float(regression_threshold)
        if regression_tolerances is not None and not isinstance(regression_tolerances, dict):
            raise TypeError("regression_tolerances must map metric names to relative tolerance values.")
        self.regression_tolerances: dict[str, float] = {}
        for name, value in (regression_tolerances or {}).items():
            if not isinstance(name, str) or not name:
                raise ValueError("Regression tolerance metric names must be non-empty strings.")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Regression tolerance for {name!r} must be finite and in [0, 1].")
            self.regression_tolerances[name] = float(value)

    def compare(self, baseline: dict[str, Any], current: dict[str, Any]) -> BenchmarkReport:
        before = validate_metrics(baseline, label="baseline metrics")
        after = validate_metrics(current, label="post metrics")
        if set(before) != set(after):
            raise ValueError("Baseline and post metric names must match exactly.")
        if set(self.directions) != set(before):
            raise ValueError(
                "Metric directions must name every compared metric exactly; "
                f"metrics={sorted(before)}, directions={sorted(self.directions)}."
            )
        unknown_tolerances = sorted(set(self.regression_tolerances) - set(before))
        if unknown_tolerances:
            raise ValueError(f"Regression tolerances name unknown metric(s): {unknown_tolerances}.")

        details: dict[str, dict[str, Any]] = {}
        regressions: list[str] = []
        for name, direction in self.directions.items():
            baseline_value = before[name]
            current_value = after[name]
            tolerance_ratio = self.regression_tolerances.get(
                name,
                DEFAULT_TOLERANCES.get(name, self.regression_threshold),
            )
            allowed = abs(baseline_value) * tolerance_ratio
            delta = current_value - baseline_value
            regressed = delta > allowed if direction == "min" else delta < -allowed
            details[name] = {
                "baseline": baseline_value,
                "post": current_value,
                "delta": delta,
                "direction": direction,
                "tolerance_ratio": tolerance_ratio,
                "allowed_absolute_delta": allowed,
                "regressed": regressed,
            }
            if regressed:
                regressions.append(name)
        return BenchmarkReport(
            per_benchmark=details,
            regressions=regressions,
            status="regression" if regressions else "passed",
        )

    def run(
        self,
        model: Any,
        tokenizer: Any,
        baseline: dict[str, Any],
        texts: list[str],
        max_length: int = 256,
        evaluator: Callable[..., dict[str, Any]] = evaluate_causal_lm,
    ) -> BenchmarkReport:
        evaluation = evaluator(model, tokenizer, texts, max_length=max_length)
        return self.compare(baseline, evaluation["metrics"])


def run(adapter: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Measure post-edit model behavior, store the comparison, then fail on regressions."""
    baseline = context.get("baseline_metrics")
    if baseline is None:
        raise ValueError(
            "No baseline_metrics are available; enable tuning.baseline_auto_pre_post "
            "or run the paired baseline stage first."
        )
    baseline = validate_metrics(baseline, label="baseline metrics")
    texts = load_evaluation_texts(context)
    max_length = int(context.get("evaluation_max_length", 256))
    evaluation = evaluate_causal_lm(
        adapter.handle.model,
        adapter.handle.tokenizer,
        texts,
        max_length=max_length,
    )
    current = validate_metrics(evaluation["metrics"], label="post metrics")
    directions = context.get("regression_metric_directions", DEFAULT_DIRECTIONS)
    tolerance = context.get("regression_tolerance_ratio", 0.02)
    tolerances = context.get("regression_tolerances")
    if not isinstance(directions, dict):
        raise TypeError("regression_metric_directions must map metric names to 'min' or 'max'.")
    checker = BenchmarkRegressionAutoTuning(
        directions=directions,
        regression_threshold=tolerance,
        regression_tolerances=tolerances,
    )
    comparison = checker.compare(baseline, current)
    report = {
        "status": comparison.status,
        "baseline_metrics": baseline,
        "post_metrics": current,
        "metrics": comparison.per_benchmark,
        "regressions": comparison.regressions,
        "tolerance_ratio": checker.regression_threshold,
        "tolerances": {
            name: checker.regression_tolerances.get(name, DEFAULT_TOLERANCES.get(name, checker.regression_threshold))
            for name in current
        },
        "sample_count": evaluation["sample_count"],
        "scored_tokens": evaluation["scored_tokens"],
        "evaluation_max_length": max_length,
    }
    context["benchmark_regression_report"] = report
    if comparison.regressions:
        raise RegressionDetectedError(
            "Post-edit benchmark regression exceeded tolerance for: "
            + ", ".join(comparison.regressions)
        )
    return report
