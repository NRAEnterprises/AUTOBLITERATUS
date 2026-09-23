"""Tuning type 82: custom benchmark auto tuning.

Run user-supplied benchmark suites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


TOGGLE = {"id": "tuning.custom_benchmark_auto", "default": True}


@dataclass
class CustomBenchmarkReport:
    per_suite: dict = field(default_factory=dict)


class CustomBenchmarkAutoTuning:
    def __init__(self, run_suite: Callable, suite_paths: list, enabled: bool = True) -> None:
        self.run_suite = run_suite
        self.suite_paths = suite_paths
        self.enabled = enabled

    def run(self, model, tokenizer) -> CustomBenchmarkReport:
        if not self.enabled:
            return CustomBenchmarkReport()
        per = {}
        for path in self.suite_paths:
            per[path] = self.run_suite(model, tokenizer, path)
        return CustomBenchmarkReport(per_suite=per)
