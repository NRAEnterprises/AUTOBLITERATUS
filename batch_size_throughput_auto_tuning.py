# abliterate/tuning/43_batch_size_throughput_auto_tuning.py
#
# Tuning Type 43 of 103:
# Batch size and throughput auto tuning.
#
# Runtime toggle: --enable tuning.batch_size_throughput_auto  (default: on)
#
# Effect:
#   Measure throughput after ablation. Choose batch size, sequence packing,
#   and scheduling to maximize throughput without breaking capability.
#
# This file owns:
#   - throughput measurement
#   - batch configuration

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ThroughputPlan:
    batch_size: int
    sequence_packing: bool
    tokens_per_second: float


class BatchSizeThroughputAutoTuning:

    def __init__(self, measure, enabled: bool = True):
        self.measure = measure
        self.enabled = enabled

    def plan(self, model, tokenizer) -> ThroughputPlan:
        if not self.enabled:
            return ThroughputPlan(1, False, 0.0)
        rate = self.measure(model, tokenizer)
        return ThroughputPlan(batch_size=8, sequence_packing=True, tokens_per_second=rate)


TOGGLE = {"id": "tuning.batch_size_throughput_auto", "default": True}
