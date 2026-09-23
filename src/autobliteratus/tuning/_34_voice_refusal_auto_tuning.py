"""Tuning type 34: voice refusal auto tuning.

For audio-capable models, compute refusal direction from voice interactions.
"""

from __future__ import annotations

from dataclasses import dataclass, field


TOGGLE = {"id": "tuning.voice_refusal_auto", "default": False}


@dataclass
class VoiceRefusalDirection:
    vector: list = field(default_factory=list)
    samples_used: int = 0


class VoiceRefusalAutoTuning:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def compute(self, model, voice_prompts: list) -> VoiceRefusalDirection:
        if not self.enabled:
            return VoiceRefusalDirection()
        return VoiceRefusalDirection(samples_used=len(voice_prompts))
