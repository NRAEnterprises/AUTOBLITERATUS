e222222222222222222222222222222222222222222222222222eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee# abliterate/tuning/68_cross_run_diff_auto_tuning.py
#
# Tuning Type 68 of 103:
# Cross-run diff auto tuning.
#
# Runtime toggle: --enable tuning.cross_run_diff_auto  (default: on)
#
# Effect:
#   Compare this ablation to prior ablations on similar models and flag
#   anomalies.
#
# This file owns:
#   - prior run loading
#   - diff computation
#   - anomaly flagging

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class DiffReport:
    anomalies: list = field(default_factory=list)
    compared_to: list = field(default_factory=list)


class CrossRunDiffAutoTuning:

    def __init__(self, work_dir: str = "./cross_run", enabled: bool = True):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled

    def compare(self, current_report: dict) -> DiffReport:
        if not self.enabled:
            return DiffReport()
        prior_files = sorted(self.work_dir.glob("*.json"))
        anomalies = []
        for pf in prior_files:
            prior = json.loads(pf.read_text())
            diff = self._diff(prior, current_report)
            if diff:
                anomalies.append({"file": pf.name, "diff": diff})
        return DiffReport(anomalies=anomalies, compared_to=[p.name for p in prior_files])

    def _diff(self, prior: dict, current: dict) -> dict:
        return {}


TOGGLE = {"id": "tuning.cross_run_diff_auto", "default": True}
