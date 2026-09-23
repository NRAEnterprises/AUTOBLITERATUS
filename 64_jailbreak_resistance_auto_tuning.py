# abliterate/tuning/64_jailbreak_resistance_auto_tuning.py
#
# Tuning Type 64 of 103:
# Jailbreak resistance auto tuning.
#
# Runtime toggle: --enable tuning.jailbreak_resistance_auto  (default: on)
#
# Effect:
#   Verify the ablated model is not trivially jailbroken by adversarial
#   prompts.
#
# This file owns:
#   - adversarial prompt generation
#   - jailbreak rate measurement
#   - correction when resistance is too low

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class JailbreakProfile:
    break_rate: float
    corrected: bool
    successful_prompts: list = field(default_factory=list)


class JailbreakResistanceAutoTuning:

    def __init__(self, measure: Callable, apply_correction: Callable,
                 enabled: bool = True, max_break_rate: float = 0.05):
        self.measure = measure
        self.apply_correction = apply_correction
        self.enabled = enabled
        self.max_break_rate = max_break_rate

    def run(self, model, tokenizer) -> JailbreakProfile:
        if not self.enabled:
            return JailbreakProfile(0.0, False)
        rate, prompts = self.measure(model, tokenizer)
        if rate > self.max_break_rate:
            self.apply_correction(model)
            return JailbreakProfile(rate, True, prompts)
        return JailbreakProfile(rate, False, prompts)


TOGGLE = {"id": "tuning.jailbreak_resistance_auto", "default": True}
