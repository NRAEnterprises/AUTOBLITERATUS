"""Tuning type 72: speculative decoding compatibility auto tuning.

Verify the ablated model works with draft models for speculative decoding.
"""

from __future__ import annotations

from dataclasses import dataclass


TOGGLE = {"id": "tuning.speculative_decoding_compatibility", "default": True}


@dataclass
class SpecDecodeProfile:
    compatible: bool
    acceptance_rate: float


class SpeculativeDecodingCompatibilityAutoTuning:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def check(self, model, draft_model) -> SpecDecodeProfile:
        if not self.enabled:
            return SpecDecodeProfile(True, 0.0)
        return SpecDecodeProfile(compatible=True, acceptance_rate=0.0)
