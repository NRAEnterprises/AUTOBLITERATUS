"""Tuning type 88: provenance recording auto tuning.

Record hashes of inputs, outputs, and weights at every stage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


TOGGLE = {"id": "tuning.provenance_recording_auto", "default": True}


@dataclass
class ProvenanceEntry:
    stage: str
    inputs_hash: str
    outputs_hash: str
    weights_hash: str


class ProvenanceRecordingAutoTuning:
    def __init__(self, work_dir: str = "./provenance", enabled: bool = True) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.entries: list[ProvenanceEntry] = []

    def record(self, stage: str, inputs, outputs, model) -> None:
        if not self.enabled:
            return
        entry = ProvenanceEntry(
            stage=stage,
            inputs_hash=self._hash(inputs),
            outputs_hash=self._hash(outputs),
            weights_hash=self._hash_model(model),
        )
        self.entries.append(entry)
        with (self.work_dir / "provenance.jsonl").open("a") as f:
            f.write(json.dumps(entry.__dict__) + "\n")

    def _hash(self, value) -> str:
        return hashlib.sha256(repr(value).encode()).hexdigest()

    def _hash_model(self, model) -> str:
        h = hashlib.sha256()
        for p in model.parameters():
            h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()
