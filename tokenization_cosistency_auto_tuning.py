# abliterate/tuning/47_tokenization_consistency_auto_tuning.py
#
# Tuning Type 47 of 103:
# Tokenization consistency auto tuning.
#
# Runtime toggle: --enable tuning.tokenization_consistency_auto  (default: on)
#
# Effect:
#   Verify the same input produces the same token sequence before and after
#   ablation.
#
# This file owns:
#   - consistency check across the corpus
#   - report of any drift

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConsistencyReport:
    drift_count: int
    samples_checked: int
    examples: list = field(default_factory=list)


class TokenizationConsistencyAutoTuning:

    def __init__(self, enabled: bool = True):
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


TOGGLE = {"id": "tuning.tokenization_consistency_auto", "default": True}
