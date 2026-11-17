from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from autobliteratus.tuning.mutation_transaction import (
    MutationTransaction,
    fingerprint_model,
)


def test_simulated_mutation_stage_failure_rolls_back_checkpoint(tmp_path) -> None:
    model = torch.nn.Linear(3, 3, bias=False)
    adapter = SimpleNamespace(handle=SimpleNamespace(model=model), modules=SimpleNamespace(model=model))
    transaction = MutationTransaction.begin(adapter, tmp_path, stage="simulated-stage")
    before = transaction.before_fingerprint

    try:
        with torch.no_grad():
            model.weight.mul_(0.0)
        raise RuntimeError("simulated downstream quality-gate failure")
    except RuntimeError as exc:
        rollback = transaction.rollback(str(exc))

    assert rollback["status"] == "rolled_back"
    assert fingerprint_model(model) == before
    manifest = json.loads((transaction.checkpoint_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "rolled_back"
    assert "simulated downstream quality-gate failure" in manifest["reason"]
