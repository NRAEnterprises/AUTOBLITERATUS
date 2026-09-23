"""Tuning type 70: version compatibility auto tuning.

Verify the ablated model loads in the target inference frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass, field


TOGGLE = {"id": "tuning.version_compatibility_auto", "default": True}


@dataclass
class CompatibilityReport:
    per_framework: dict = field(default_factory=dict)


class VersionCompatibilityAutoTuning:
    def __init__(self, loaders: dict, enabled: bool = True) -> None:
        self.loaders = loaders
        self.enabled = enabled

    def run(self, model_path: str) -> CompatibilityReport:
        if not self.enabled:
            return CompatibilityReport()
        results = {}
        for name, loader in self.loaders.items():
            try:
                loader(model_path)
                results[name] = "ok"
            except Exception as e:
                results[name] = f"fail: {e}"
        return CompatibilityReport(per_framework=results)
