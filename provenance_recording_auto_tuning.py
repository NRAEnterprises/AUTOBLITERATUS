# abliterate/tuning/88_provenance_recording_auto_tuning.py
#
# Tuning Type 88 of 103:
# Provenance recording auto tuning.
#
# Runtime toggle: --enable tuning.provenance_recording_auto  (default: on)
#
# Effect:
#   Record hashes of inputs, outputs, and weights at every stage.
#
# This file owns:
#   - hash capture per stage
#   - provenance file writing

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json


@dataclass
class ProvenanceEntry:
    stage: str
    inputs_hash: str
    outputs_hash: str
    weights_hash: str


class ProvenanceRecordingAutoTuning:

    def __init__(self, work_dir: str = "./provenance", enabled: bool = True):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.entries: list = []

    def record(self, stage: str, inputs, outputs, model):
        if not self.enabled:
            return
        entry = ProvenanceEntry(
            stage=stage,
            inputs_hash=self._hash(inputs),
            outputs_hash=self._hash(outputs),
            weights_hash=self._hash_model(model),
        )
        self.entries.append(entry)
        (self.work_dir / "provenance.jsonl").open("a").write(
            json.dumps(entry.__dict__) + "\n")

    def _hash(self, value) -> str:
        return hashlib.sha256(repr(value).encode()).hexdigest()

    def _hash_model(self, model) -> str:
        h = hashlib.sha256()
        for p in model.parameters():
            h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()


TOGGLE = {"id": "tuning.provenance_recording_auto", "default": True}
