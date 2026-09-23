# abliterate/tuning/12_telemetry_and_threshold_auto_tuning.py
#
# Tuning Type 12 of 103:
# Telemetry and threshold auto tuning.
#
# Runtime toggle: --enable tuning.telemetry_and_threshold_auto  (default: on)
#
# Effect:
#   Derive thresholds from the model's own numbers. Learn from the population
#   of prior runs. Every threshold is a function of the model's baseline, not
#   a constant.
#
# This file owns:
#   - the telemetry knowledge base keyed by (arch_class, reasoning_class, param_bucket)
#   - fall-through to broader buckets when exact key missing
#   - threshold derivation from the baseline
#
# This file consumes:
#   - the baseline pair (from tuning/07, passed in)
#   - prior runs from the knowledge base

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TelemetryEntry:
    arch_class: str
    reasoning_class: str
    param_bucket: str
    thresholds: dict = field(default_factory=dict)


class TelemetryAndThresholdAutoTuning:

    def __init__(self, work_dir: str = "./abliterate_workspace", enabled: bool = True):
        self.work_dir = Path(work_dir)
        self.enabled = enabled
        self.kb_path = self.work_dir / "telemetry_kb.json"

    def derive(self, baseline_pair, arch_class: str, reasoning_class: str,
               param_bucket: str) -> dict:
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

    def _lookup(self, arch_class, reasoning_class, param_bucket):
        if not self.kb_path.exists():
            return None
        import json
        kb = json.loads(self.kb_path.read_text())
        key = f"{arch_class}|{reasoning_class}|{param_bucket}"
        if key in kb:
            return TelemetryEntry(arch_class, reasoning_class, param_bucket, kb[key])
        broader = f"{arch_class}|*|*"
        if broader in kb:
            return TelemetryEntry(arch_class, reasoning_class, param_bucket, kb[broader])
        return None


TOGGLE = {"id": "tuning.telemetry_and_threshold_auto", "default": True}
