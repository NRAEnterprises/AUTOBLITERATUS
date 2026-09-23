"""Tuning type 66: weight fingerprint auto tuning.

Record the model's weight fingerprint at each stage so rollback can verify
integrity.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


TOGGLE = {"id": "tuning.weight_fingerprint_auto", "default": True}


@dataclass
class Fingerprint:
    stage: str
    digest: str


class WeightFingerprintAutoTuning:
    def __init__(self, work_dir: str = "./fingerprints", enabled: bool = True) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled

    def compute(self, model, stage: str) -> Fingerprint:
        if not self.enabled:
            return Fingerprint(stage, "")
        h = hashlib.sha256()
        for p in model.parameters():
            h.update(p.detach().cpu().numpy().tobytes())
        digest = h.hexdigest()
        (self.work_dir / f"{stage}.fp").write_text(digest)
        return Fingerprint(stage, digest)

    def verify(self, model, stage: str) -> bool:
        if not self.enabled:
            return True
        expected = self.work_dir / f"{stage}.fp"
        if not expected.exists():
            return False
        return expected.read_text() == self.compute(model, stage).digest
