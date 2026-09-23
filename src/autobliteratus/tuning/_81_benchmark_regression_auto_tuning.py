"""Tuning type 81: benchmark regression auto tuning.

Run standard benchmarks before and after ablation. Flag regressions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


TOGGLE = {"id": "tuning.benchmark_regression_auto", "default": True}


@dataclass
class BenchmarkReport:
    per_benchmark: dict = field(default_factory=dict)
    regressions: list = field(default_factory=list)


class BenchmarkRegressionAutoTuning:
    def __init__(
        self,
        run_benchmark: Callable,
        benchmarks: list,
        enabled: bool = True,
        regression_threshold: float = 0.03,
    ) -> None:
        self.run_benchmark = run_benchmark
        self.benchmarks = benchmarks
        self.enabled = enabled
        self.regression_threshold = regression_threshold

    def run(self, model, tokenizer, baseline: dict) -> BenchmarkReport:
        if not self.enabled:
            return BenchmarkReport()
        per = {}
        regressions = []
        for bench in self.benchmarks:
            score = self.run_benchmark(model, tokenizer, bench)
            per[bench] = score
            if score < baseline.get(bench, 0.0) - self.regression_threshold:
                regressions.append(bench)
        return BenchmarkReport(per_benchmark=per, regressions=regressions)
