"""Tuning type 67: reproducibility auto tuning.

Record seeds, RNG state, and environment so the same ablation can be
reproduced byte-for-byte.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path


TOGGLE = {"id": "tuning.reproducibility_auto", "default": True}


@dataclass
class ReproducibilityRecord:
    seeds: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)


class ReproducibilityAutoTuning:
    def __init__(self, work_dir: str = "./reproducibility", enabled: bool = True) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled

    def capture(self) -> ReproducibilityRecord:
        if not self.enabled:
            return ReproducibilityRecord()
        seeds = {
            "python": random.getstate()[1][0],
            "numpy": self._numpy_seed(),
            "torch": self._torch_seed(),
        }
        env = {k: v for k, v in os.environ.items() if k.startswith(("CUDA", "TORCH", "PYTHON"))}
        record = ReproducibilityRecord(seeds=seeds, env=env)
        (self.work_dir / "record.json").write_text(json.dumps(
            {"seeds": seeds, "env": env}, indent=2))
        return record

    def restore(self, record: ReproducibilityRecord) -> None:
        if not self.enabled:
            return
        random.seed(record.seeds.get("python", 0))
        try:
            import numpy as np
            np.random.seed(record.seeds.get("numpy", 0))
        except ImportError:
            pass
        try:
            import torch
            torch.manual_seed(record.seeds.get("torch", 0))
        except ImportError:
            pass

    def _numpy_seed(self) -> int:
        try:
            import numpy as np
            return int(np.random.get_state()[1][0])
        except ImportError:
            return 0

    def _torch_seed(self) -> int:
        try:
            import torch
            return int(torch.initial_seed())
        except ImportError:
            return 0
