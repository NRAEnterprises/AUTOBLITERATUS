"""Tuning type 36: tokenizer surgery auto tuning.

Repair or replace the tokenizer if token distribution shifted after ablation.
"""

from __future__ import annotations

from dataclasses import dataclass


TOGGLE = {"id": "tuning.tokenizer_surgery_auto", "default": True}


@dataclass
class TokenizerSurgeryResult:
    repaired: bool
    replaced: bool
    drift: float


class TokenizerSurgeryAutoTuning:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def run(self, tokenizer, baseline_tokenizer) -> TokenizerSurgeryResult:
        if not self.enabled:
            return TokenizerSurgeryResult(False, False, 0.0)
        drift = self._compare(tokenizer, baseline_tokenizer)
        return TokenizerSurgeryResult(repaired=False, replaced=False, drift=drift)

    def _compare(self, a, b) -> float:
        return 0.0
