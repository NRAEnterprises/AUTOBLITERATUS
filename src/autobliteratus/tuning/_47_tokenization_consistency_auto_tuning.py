"""Tuning type 47: tokenization consistency auto tuning.

Verify the same input produces the same token sequence before and after
ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


TOGGLE = {"id": "tuning.tokenization_consistency_auto", "default": True}


@dataclass
class ConsistencyReport:
    drift_count: int
    samples_checked: int
    examples: list = field(default_factory=list)


class TokenizationConsistencyAutoTuning:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def run(self, tokenizer_before, tokenizer_after, corpus: list) -> ConsistencyReport:
        if not self.enabled:
            return ConsistencyReport(0, 0)
        drift = 0
        examples = []
        for sample in corpus:
            a = tokenizer_before.encode(sample)
            b = tokenizer_after.encode(sample)
            if a != b:
                drift += 1
                if len(examples) < 5:
                    examples.append(sample)
        return ConsistencyReport(drift_count=drift, samples_checked=len(corpus), examples=examples)
