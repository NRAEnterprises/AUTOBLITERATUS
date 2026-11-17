from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from autobliteratus.core.topology_proposal import (
    SAFETY_CONTRACT,
    TopologyProposalError,
    validate_topology_proposal,
)
from autobliteratus.tuning.mutation_firewall import (
    MutationAuthorizationError,
    require_mutation_authorized,
)
from autobliteratus.tuning.mutation_transaction import (
    MutationTransaction,
    fingerprint_model,
)


def _proposal(identity: dict[str, str]) -> dict:
    return {
        "schema_version": 1,
        "export_kind": "model_mapping_observatory_topology",
        "generated_at": "2026-11-17T00:00:00Z",
        "model_identity": identity,
        "safety": dict(SAFETY_CONTRACT),
        "viewport": {
            "view": "graph",
            "selected_node_id": "root",
            "filter": {"query": "", "evidence": "all"},
        },
        "nodes": [{"id": "root", "label": "Model", "status": "verified"}],
        "edges": [],
        "mind_map": {"root_id": "root", "branch_ids": [], "interchange": []},
        "backend_context": [],
    }


def test_mutation_firewall_is_fail_closed() -> None:
    with pytest.raises(MutationAuthorizationError):
        require_mutation_authorized("tuning.refusal_direction_ablation", {})


def test_transaction_rolls_back_exact_model_state(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    before = copy.deepcopy(model.state_dict())
    adapter = SimpleNamespace(handle=SimpleNamespace(model=model), modules=SimpleNamespace(model=model))
    transaction = MutationTransaction.begin(adapter, tmp_path, model_revision="rev-1")
    checkpoint_root = tmp_path / "mutation_checkpoints"
    assert transaction.checkpoint_dir.parent == checkpoint_root
    assert not any(path.name.startswith(".") for path in checkpoint_root.iterdir())
    with torch.no_grad():
        model.weight.add_(1.0)
    assert fingerprint_model(model) != transaction.before_fingerprint
    report = transaction.rollback("test_failure")
    assert report["status"] == "rolled_back"
    assert transaction.rolled_back is True
    assert fingerprint_model(model) == transaction.before_fingerprint
    for name, value in before.items():
        assert torch.equal(model.state_dict()[name], value)
    checkpoint = json.loads((transaction.checkpoint_dir / "checkpoint.json").read_text())
    assert checkpoint["status"] == "rolled_back"


def test_transaction_commit_requires_a_real_change(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    adapter = SimpleNamespace(handle=SimpleNamespace(model=model), modules=SimpleNamespace(model=model))
    transaction = MutationTransaction.begin(adapter, tmp_path)
    with pytest.raises(RuntimeError, match="no state change"):
        transaction.commit()
    transaction.rollback("cleanup")


def test_topology_proposal_binds_identity_and_is_non_mutating() -> None:
    identity = {"name": "fixture", "revision": "rev-1", "architecture": "decoder-only"}
    report = validate_topology_proposal(_proposal(identity), expected_identity=identity)
    assert report["validated"] is True
    assert report["direct_weight_mutation"] is False
    assert report["mutation_authorized"] is False


def test_topology_proposal_rejects_identity_and_safety_tampering() -> None:
    identity = {"name": "fixture", "revision": "rev-1", "architecture": "decoder-only"}
    expected_identity = dict(identity)
    payload = _proposal(identity)
    payload["model_identity"]["revision"] = "stale"
    with pytest.raises(TopologyProposalError, match="identity mismatch"):
        validate_topology_proposal(payload, expected_identity=expected_identity)

    payload = _proposal(identity)
    payload["safety"]["direct_weight_mutation"] = True
    with pytest.raises(TopologyProposalError, match="safety contract"):
        validate_topology_proposal(payload, expected_identity=expected_identity)


def test_topology_proposal_rejects_unknown_edge_endpoint() -> None:
    identity = {"name": "fixture", "revision": "rev-1", "architecture": "decoder-only"}
    payload = _proposal(identity)
    payload["edges"] = [{"source": "root", "target": "missing"}]
    with pytest.raises(TopologyProposalError, match="unknown node"):
        validate_topology_proposal(payload, expected_identity=identity)
