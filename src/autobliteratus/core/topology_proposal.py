"""Validate observatory topology exports before downstream mutation use."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EXPORT_KIND = "model_mapping_observatory_topology"
SAFETY_CONTRACT = {
    "direct_weight_mutation": False,
    "downstream_mode": "sandboxed_proposal",
    "requires_pipeline_validation": True,
}


class TopologyProposalError(ValueError):
    """Raised when a topology export is unsafe or mismatched."""


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_payload(source: str | os.PathLike[str] | Mapping[str, Any], sandbox_root: Path | None) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source).expanduser().resolve()
    if sandbox_root is not None:
        root = sandbox_root.expanduser().resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise TopologyProposalError("Topology proposal must remain inside sandbox_root.") from exc
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TopologyProposalError(f"Topology proposal is not readable JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise TopologyProposalError("Topology proposal root must be an object.")
    return payload


def _validate_graph(payload: Mapping[str, Any]) -> None:
    nodes = payload.get("nodes")
    edges = payload.get("edges")
    if not isinstance(nodes, list) or not nodes:
        raise TopologyProposalError("Topology proposal requires a non-empty nodes array.")
    if not isinstance(edges, list):
        raise TopologyProposalError("Topology proposal requires an edges array.")
    node_ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or not node["id"]:
            raise TopologyProposalError("Every topology node must have a non-empty string id.")
        if node["id"] in node_ids:
            raise TopologyProposalError(f"Duplicate topology node id: {node['id']}")
        node_ids.add(node["id"])
    for edge in edges:
        if not isinstance(edge, dict):
            raise TopologyProposalError("Every topology edge must be an object.")
        source = edge.get("source")
        target = edge.get("target")
        if source not in node_ids or target not in node_ids:
            raise TopologyProposalError("Topology edge references an unknown node.")


def validate_topology_proposal(
    source: str | os.PathLike[str] | Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    sandbox_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return a safe proposal envelope bound to the expected model identity.

    The returned object deliberately contains no callable operation and always
    reports ``direct_weight_mutation=False``. A caller must perform separate
    runtime mapping and quality-gate checks before proposing any edit.
    """
    root = Path(sandbox_root).expanduser().resolve() if sandbox_root is not None else None
    payload = _load_payload(source, root)
    if payload.get("schema_version") != 1:
        raise TopologyProposalError("Unsupported topology schema_version.")
    if payload.get("export_kind") != EXPORT_KIND:
        raise TopologyProposalError("Unsupported topology export_kind.")
    if payload.get("safety") != SAFETY_CONTRACT:
        raise TopologyProposalError("Topology safety contract is missing or has been altered.")
    identity = payload.get("model_identity")
    if not isinstance(identity, dict):
        raise TopologyProposalError("Topology proposal requires a model_identity object.")
    for key, expected in expected_identity.items():
        if key not in identity or identity[key] != expected:
            raise TopologyProposalError(f"Topology model identity mismatch for {key!r}.")
    _validate_graph(payload)
    viewport = payload.get("viewport")
    if not isinstance(viewport, dict) or viewport.get("selected_node_id") not in {
        node["id"] for node in payload["nodes"]
    }:
        raise TopologyProposalError("Topology viewport selected_node_id must reference a known node.")
    digest = _canonical_digest(payload)
    return {
        "validated": True,
        "schema_version": 1,
        "export_kind": EXPORT_KIND,
        "model_identity": dict(identity),
        "proposal_digest": digest,
        "node_ids": sorted(node["id"] for node in payload["nodes"]),
        "edge_count": len(payload["edges"]),
        "safety": dict(SAFETY_CONTRACT),
        "direct_weight_mutation": False,
        "downstream_mode": "sandboxed_proposal",
        "mutation_authorized": False,
    }
