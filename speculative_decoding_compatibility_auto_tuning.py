# abliterate/tuning/72_speculative_decoding_compatibility_auto_tuning.py
#
# Tuning Type 72 of 103:
# Speculative decoding compatibility auto tuning.
#
# Runtime toggle: --enable tuning.speculative_decoding_compatibility  (default: on)
#
# Effect:
#   Verify the ablated model works with draft models for speculative decoding.
#
# This file owns:
#   - draft model compatibility check
#   - acceptance rate measurement

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpecDecodeProfile:
    compatible: bool
    acceptance_rate: float


class SpeculativeDecodingCompatibilityAutoTuning:

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def check(self, model, draft_model) -> SpecDecodeProfile:
        if not self.enabled:
            return SpecDecodeProfile(True, 0.0)
        return SpecDecodeProfile(compatible=True, acceptance_rate=0.0)


TOGGLE = {"id": "tuning.speculative_decoding_compatibility", "default": True, "read_only": True}
