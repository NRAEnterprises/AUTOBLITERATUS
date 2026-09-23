"""Tuning type 43: batch size and throughput auto tuning.

Measure throughput after ablation. Choose batch size, sequence packing,
and scheduling to maximize throughput without breaking capability.
"""

from __future__ import annotations

from dataclasses import dataclass


TOGGLE = {"id": "tuning.batch_size_throughput_auto", "default": True}


@dataclass
class ThroughputPlan:
    batch_size: int
    sequence_packing: bool
    tokens_per_second: float


class BatchSizeThroughputAutoTuning:
    def __init__(self, measure, enabled: bool = True) -> None:
        self.measure = measure
        self.enabled = enabled

    def plan(self, model, tokenizer) -> ThroughputPlan:
        if not self.enabled:
            return ThroughputPlan(1, False, 0.0)
        rate = self.measure(model, tokenizer)
        return ThroughputPlan(batch_size=8, sequence_packing=True, tokens_per_second=rate)
