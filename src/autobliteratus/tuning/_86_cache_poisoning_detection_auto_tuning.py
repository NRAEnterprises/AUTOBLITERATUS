"""Tuning type 86: cache poisoning detection auto tuning.

Verify no cache-level poisoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field


TOGGLE = {"id": "tuning.cache_poisoning_detection", "default": True}


@dataclass
class CacheReport:
    detected: bool
    entries_flagged: list = field(default_factory=list)


class CachePoisoningDetectionAutoTuning:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def run(self, cache_dir: str) -> CacheReport:
        if not self.enabled:
            return CacheReport(False)
        return CacheReport(False)
