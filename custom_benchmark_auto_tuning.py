# abliterate/tuning/82_custom_benchmark_auto_tuning.py
#
# Tuning Type 82 of 103:
# Custom benchmark auto tuning.
#
# Runtime toggle: --enable tuning.custom_benchmark_auto  (default: on)
#
# Effect:
#   Run user-supplied benchmark suites.
#
# This file owns:
#   - loading user benchmark definitions
#   - running them
#   - reporting results

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class CustomBenchmarkReport:
    per_suite: dict = field(default_factory=dict)


class CustomBenchmarkAutoTuning:

    def __init__(self, run_suite: Callable, suite_paths: list,
                 enabled: bool = True):
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


TOGGLE = {"id": "tuning.custom_benchmark_auto", "default": True}
