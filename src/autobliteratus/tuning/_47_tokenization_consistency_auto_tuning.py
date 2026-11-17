"""Tuning type 47: tokenization consistency auto tuning.

Verify the same input produces the same token sequence before and after
ablation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

TOGGLE = {
    "id": "tuning.tokenization_consistency_auto",
    "default": False,
    "desc": "Compare token IDs from explicitly supplied before/after tokenizers without mutating the model.",
}


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


def run(adapter: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Compare caller-supplied baseline and current tokenizers on a finite corpus."""
    del adapter
    before = context.get("tokenizer_before")
    after = context.get("tokenizer_after")
    corpus = context.get("tokenization_corpus", [])
    max_samples = context.get("tokenization_max_samples", 10_000)
    if isinstance(max_samples, bool) or not isinstance(max_samples, int) or not 1 <= max_samples <= 10_000:
        raise ValueError("tokenization_max_samples must be an integer in [1, 10000].")
    if before is None or after is None:
        report = {
            "status": "not_evaluated",
            "reason": "baseline_and_current_tokenizers_required",
            "model_modified": False,
        }
    else:
        if not isinstance(corpus, (list, tuple)) or len(corpus) > max_samples or not all(isinstance(text, str) for text in corpus):
            raise TypeError("tokenization_corpus must be a list of strings.")
        if any(len(text) > 1_000_000 for text in corpus):
            raise ValueError("tokenization_corpus entries must be at most 1000000 characters.")
        rows = []
        drift_count = 0
        for text in corpus:
            before_ids = list(before.encode(text))
            after_ids = list(after.encode(text))
            if before_ids != after_ids:
                drift_count += 1
                if len(rows) < 5:
                    rows.append(
                        {
                            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                            "text_length": len(text),
                            "before_token_count": len(before_ids),
                            "after_token_count": len(after_ids),
                        }
                    )
        report = {
            "status": "measured",
            "drift_count": drift_count,
            "samples_checked": len(corpus),
            "drift_rate": drift_count / len(corpus) if corpus else 0.0,
            "examples": rows,
            "model_modified": False,
        }
    context["tokenization_consistency_report"] = report
    return report
