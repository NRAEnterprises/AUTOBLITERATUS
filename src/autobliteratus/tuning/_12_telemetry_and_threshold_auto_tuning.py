"""Tuning type 12: telemetry and threshold auto tuning.

Derive thresholds from the model's own numbers and the population of prior
runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


TOGGLE = {"id": "tuning.telemetry_and_threshold_auto", "default": True}


@dataclass
class TelemetryEntry:
    arch_class: str
    reasoning_class: str
    param_bucket: str
    thresholds: dict = field(default_factory=dict)


class TelemetryAndThresholdAutoTuning:
    def __init__(self, work_dir: str = "./abliterate_workspace", enabled: bool = True) -> None:
        self.work_dir = Path(work_dir)
        self.enabled = enabled
        self.kb_path = self.work_dir / "telemetry_kb.json"

    def derive(
        self,
        baseline_pair,
        arch_class: str,
        reasoning_class: str,
        param_bucket: str,
    ) -> dict:
        if not self.enabled:
            return {}
        thresholds = self._from_baseline(baseline_pair)
        kb_entry = self._lookup(arch_class, reasoning_class, param_bucket)
        if kb_entry:
            for k, v in kb_entry.thresholds.items():
                thresholds.setdefault(k, v)
        return thresholds

    def _from_baseline(self, baseline_pair) -> dict:
        out = {}
        for axis, base in baseline_pair.pre.items():
            out[f"{axis}_min"] = base * 0.98
        return out

    def _lookup(self, arch_class, reasoning_class, param_bucket) -> TelemetryEntry | None:
        if not self.kb_path.exists():
            return None
        kb = json.loads(self.kb_path.read_text())
        key = f"{arch_class}|{reasoning_class}|{param_bucket}"
        if key in kb:
            return TelemetryEntry(arch_class, reasoning_class, param_bucket, kb[key])
        broader = f"{arch_class}|*|*"
        if broader in kb:
            return TelemetryEntry(arch_class, reasoning_class, param_bucket, kb[broader])
        return None
