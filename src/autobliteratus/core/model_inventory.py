"""Immutable model inventory and evidence-only role discovery.

This module records what a scanner observed. It does not authorize model edits.
Framework-specific execution, isolation, task metrics, and post-removal recovery
are supplied by explicit adapters; missing evidence remains unknown.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sqlite3
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
_SCHEMA_PATH = Path(__file__).resolve().parent / "model_inventory_schema_v1.sql"
_DETECTOR_NAME = "graph-evidence-role-inference"
_DETECTOR_VERSION = "0.1.0"
TAXONOMY_VERSION = "model-role/v1"

# Each profile is run for every non-root registered module. Adapter output must
# contain profile-specific, numeric/structured measurements in addition to an
# explicit isolated_copy=true attestation.
DEFAULT_PROBE_PROFILES = (
    "output_sensitivity",
    "capability_axes",
    "input_regime_robustness",
    "resource_impact",
    "stability_recovery",
)
_REQUIRED_AXES = ("reasoning", "general", "cohesion")
_SECRET_FIELD = re.compile(r"(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|secret|credential|private[_-]?key)", re.IGNORECASE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(prefix: str, value: Any) -> str:
    return f"{prefix}:v1:" + hashlib.sha256((prefix + ":v1\0" + canonical_json(value)).encode("utf-8")).hexdigest()


def _safe_json(value: Any) -> Any:
    """Convert evidence to JSON values while refusing opaque object reprs."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items() if not _SECRET_FIELD.search(str(key))}
    if isinstance(value, (tuple, list)):
        return [_safe_json(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _safe_json(value.item())
        except (TypeError, ValueError, RuntimeError):
            return None
    return str(value) if isinstance(value, Path) else None


def _type_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(dim) for dim in shape]
    except (TypeError, ValueError):
        return None


def _tensor_bytes(value: Any) -> bytes | None:
    candidate = value
    if callable(getattr(candidate, "detach", None)):
        candidate = candidate.detach()
    if callable(getattr(candidate, "cpu", None)):
        candidate = candidate.cpu()
    if callable(getattr(candidate, "contiguous", None)):
        candidate = candidate.contiguous()
    if callable(getattr(candidate, "numpy", None)):
        try:
            return candidate.numpy().tobytes(order="C")
        except (TypeError, ValueError, RuntimeError):
            try:
                import torch

                return candidate.view(torch.uint8).numpy().tobytes(order="C")
            except (ImportError, AttributeError, TypeError, ValueError, RuntimeError):
                return None
    if callable(getattr(candidate, "tobytes", None)):
        try:
            return candidate.tobytes(order="C")
        except TypeError:
            try:
                return candidate.tobytes()
            except (TypeError, ValueError):
                return None
    if isinstance(candidate, (str, int, float, bool, list, tuple)):
        try:
            return canonical_json(candidate).encode("utf-8")
        except (TypeError, ValueError):
            return None
    return None


def _tensor_metadata(value: Any) -> dict[str, Any]:
    shape = _shape(value)
    dtype = str(getattr(value, "dtype", type(value).__name__))
    layout = str(getattr(value, "layout", "unknown"))
    strides = getattr(value, "stride", None)
    if callable(strides):
        try:
            strides = list(strides())
        except (TypeError, RuntimeError):
            strides = None
    elif strides is not None:
        try:
            strides = list(strides)
        except TypeError:
            strides = None
    if strides is not None:
        strides = [int(item) for item in strides]
    count = getattr(value, "numel", None)
    if callable(count):
        try:
            count = int(count())
        except (TypeError, ValueError, RuntimeError):
            count = None
    elif shape is not None:
        count = math.prod(shape)
    return {"shape": shape, "dtype": dtype, "layout": layout, "strides": strides, "element_count": count}


def _enumerate_named(model: Any, method_name: str, **options: Any) -> list[tuple[str, Any]]:
    method = getattr(model, method_name, None)
    if not callable(method):
        return []
    try:
        return list(method(remove_duplicate=False, **options))
    except TypeError:
        try:
            return list(method(**options))
        except TypeError:
            try:
                return list(method())
            except (TypeError, RuntimeError):
                return []
        except RuntimeError:
            return []
    except (RuntimeError, ValueError):
        return []


def _model_config(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None:
        return {}
    if callable(getattr(config, "to_dict", None)):
        raw = config.to_dict()
    elif hasattr(config, "__dict__"):
        raw = vars(config)
    elif isinstance(config, Mapping):
        raw = config
    else:
        return {"config_type": _type_name(config)}
    return _safe_json(raw) if isinstance(raw, Mapping) else {}


def capture_inventory(
    model: Any,
    *,
    source_revision: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture module/state structure with deterministic snapshot-scoped IDs.

    Captured values are never persisted. Tensor hashes are computed in memory;
    unsupported tensor representations are explicitly marked unavailable and
    make the inventory partial.
    """
    capture_started = time.monotonic()
    raw_modules = _enumerate_named(model, "named_modules")
    module_inventory_available = bool(raw_modules)
    if not raw_modules:
        raw_modules = [("", model)]
    grouped: dict[int, dict[str, Any]] = {}
    for path, module in raw_modules:
        record = grouped.setdefault(id(module), {"object": module, "paths": []})
        record["paths"].append(str(path))
    module_groups = []
    for record in grouped.values():
        aliases = sorted(set(record["paths"]))
        canonical_path = aliases[0]
        module = record["object"]
        children = _enumerate_named(module, "named_children")
        direct_parameter_names = [name for name, _ in _enumerate_named(module, "named_parameters", recurse=False)]
        direct_buffer_names = [name for name, _ in _enumerate_named(module, "named_buffers", recurse=False)]
        module_groups.append({
            "canonical_path": canonical_path,
            "aliases": aliases,
            "module_type": _type_name(module),
            "child_count": len(children),
            "is_leaf": len(children) == 0,
            "signature": {
                "module_type": _type_name(module),
                "child_count": len(children),
                "direct_parameter_names": sorted(direct_parameter_names),
                "direct_buffer_names": sorted(direct_buffer_names),
            },
            "object": module,
        })
    module_groups.sort(key=lambda row: (row["canonical_path"].count("."), row["canonical_path"], row["module_type"]))

    named_parameters = dict(_enumerate_named(model, "named_parameters"))
    named_buffers = dict(_enumerate_named(model, "named_buffers"))
    state_method = getattr(model, "state_dict", None)
    try:
        state_dict = state_method() if callable(state_method) else {}
    except (TypeError, RuntimeError, ValueError):
        state_dict = {}
    if not isinstance(state_dict, Mapping):
        state_dict = {}

    tensor_rows: list[dict[str, Any]] = []
    partial = not module_inventory_available or not callable(state_method)
    object_to_state_names: dict[int, list[str]] = defaultdict(list)
    for name, value in named_parameters.items():
        object_to_state_names[id(value)].append(str(name))
    for name, value in named_buffers.items():
        object_to_state_names[id(value)].append(str(name))

    for name, value in sorted(state_dict.items(), key=lambda item: str(item[0])):
        name = str(name)
        detail = _tensor_metadata(value)
        raw_bytes = _tensor_bytes(value)
        value_hash = hashlib.sha256(raw_bytes).hexdigest() if raw_bytes is not None else None
        if value_hash is None:
            partial = True
        storage_alias = None
        aliases = object_to_state_names.get(id(value), [])
        if len(aliases) > 1:
            storage_alias = _digest("alias", sorted(aliases))
        kind = "parameter" if name in named_parameters else "buffer" if name in named_buffers else "state_dict"
        row = {
            "state_name": name,
            **detail,
            "value_sha256": value_hash,
            "hash_status": "verified" if value_hash is not None else "unavailable",
            "storage_alias_group": storage_alias,
            "state_kind": kind,
        }
        tensor_rows.append(row)

    config = _model_config(model)
    safe_metadata = _safe_json(dict(metadata or {}))
    if not isinstance(safe_metadata, dict):
        safe_metadata = {}
    model_type = str(config.get("model_type", "unknown"))
    structure_manifest = {
        "config": config,
        "modules": [
            {key: row[key] for key in ("canonical_path", "aliases", "module_type", "child_count", "is_leaf", "signature")}
            for row in module_groups
        ],
        "tensors": [
            {key: row[key] for key in ("state_name", "shape", "dtype", "layout", "strides", "state_kind", "storage_alias_group")}
            for row in tensor_rows
        ],
    }
    structure_id = _digest("structure", structure_manifest)
    snapshot_manifest = {
        "schema_version": SCHEMA_VERSION,
        "structure_id": structure_id,
        "source_revision": source_revision,
        "model_class": _type_name(model),
        "model_type": model_type,
        "config_sha256": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
        "metadata": safe_metadata,
        "module_inventory_available": module_inventory_available,
        "state_dict_available": callable(state_method),
        "tensor_values": [{"name": row["state_name"], "sha256": row["value_sha256"], "hash_status": row["hash_status"]} for row in tensor_rows],
    }
    snapshot_id = _digest("snapshot", snapshot_manifest)

    module_ids: dict[str, str] = {}
    for row in module_groups:
        row["module_id"] = _digest("module", {
            "snapshot_id": snapshot_id,
            "canonical_path": row["canonical_path"],
            "module_type": row["module_type"],
            "signature": row["signature"],
        })
        for alias in row["aliases"]:
            module_ids[alias] = row["module_id"]

    module_rows = []
    for row in module_groups:
        path = row["canonical_path"]
        parent_path = path.rsplit(".", 1)[0] if "." in path else ""
        parent_id = module_ids.get(parent_path) if path else None
        module_rows.append({key: row[key] for key in ("module_id", "canonical_path", "aliases", "module_type", "signature", "child_count", "is_leaf")} | {"parent_module_id": parent_id})

    tensor_name_to_id: dict[str, str] = {}
    for row in tensor_rows:
        row["tensor_id"] = _digest("tensor", {
            "snapshot_id": snapshot_id,
            "state_name": row["state_name"],
            "shape": row["shape"],
            "dtype": row["dtype"],
        })
        tensor_name_to_id[row["state_name"]] = row["tensor_id"]

    bindings = []
    module_paths = sorted(module_ids, key=lambda path: (-len(path), path))
    for tensor in tensor_rows:
        state_name = tensor["state_name"]
        owner = next((path for path in module_paths if path and (state_name == path or state_name.startswith(path + "."))), "")
        local_name = state_name[len(owner) + 1:] if owner else state_name
        if owner in module_ids:
            bindings.append({
                "module_id": module_ids[owner],
                "tensor_id": tensor_name_to_id[state_name],
                "local_name": local_name,
                "relation": tensor["state_kind"] if tensor["state_kind"] in {"parameter", "buffer"} else "state_dict",
            })

    manifest = {
        **snapshot_manifest,
        "snapshot_id": snapshot_id,
        "captured_at": _now(),
        "capture_duration_seconds": time.monotonic() - capture_started,
        "status": "partial" if partial else "complete",
        "structure_manifest": structure_manifest,
        "module_count": len(module_rows),
        "tensor_count": len(tensor_rows),
        "module_rows": module_rows,
        "tensor_rows": tensor_rows,
        "bindings": bindings,
    }
    return manifest


def _connect(database_path: str | Path) -> sqlite3.Connection:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def initialize_database(database_path: str | Path) -> Path:
    """Create v1 inventory DB or reject an unsupported newer schema."""
    if not _SCHEMA_PATH.is_file():
        raise FileNotFoundError(f"Model inventory schema not found: {_SCHEMA_PATH}")
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema_text = _SCHEMA_PATH.read_text(encoding="utf-8")
    schema_hash = hashlib.sha256(schema_text.encode("utf-8")).hexdigest()
    connection = _connect(path)
    try:
        connection.executescript(schema_text)
        row = connection.execute("SELECT schema_version, schema_sha256 FROM schema_versions ORDER BY schema_version DESC LIMIT 1").fetchone()
        current = row["schema_version"] if row else None
        if current is not None and int(current) != SCHEMA_VERSION:
            raise RuntimeError(f"Inventory database schema {current} requires an explicit migration to supported schema {SCHEMA_VERSION}.")
        if row is not None and row["schema_sha256"] != schema_hash:
            raise RuntimeError("Inventory schema contents differ from the recorded v1 schema; bump the schema version and migrate explicitly.")
        connection.execute(
            "INSERT OR IGNORE INTO schema_versions(schema_version, applied_at, schema_sha256) VALUES (?, ?, ?)",
            (SCHEMA_VERSION, _now(), schema_hash),
        )
        connection.commit()
    finally:
        connection.close()
    return path


def persist_inventory(database_path: str | Path, inventory: Mapping[str, Any]) -> str:
    """Append an immutable inventory snapshot and its normalized child rows."""
    initialize_database(database_path)
    snapshot_id = str(inventory["snapshot_id"])
    connection = _connect(database_path)
    try:
        exists = connection.execute("SELECT manifest_json FROM inventory_snapshots WHERE snapshot_id = ?", (snapshot_id,)).fetchone()
        if exists is not None:
            existing = json.loads(exists["manifest_json"])
            if existing.get("structure_id") != inventory.get("structure_id"):
                raise ValueError("Snapshot ID collision with inconsistent inventory content.")
            return snapshot_id
        compact = {
            key: inventory.get(key)
            for key in ("schema_version", "structure_id", "source_revision", "model_class", "model_type", "config_sha256", "metadata", "module_inventory_available", "state_dict_available", "tensor_values", "module_count", "tensor_count", "capture_duration_seconds", "status")
        }
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT INTO inventory_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot_id,
                str(inventory["structure_id"]),
                SCHEMA_VERSION,
                str(inventory["captured_at"]),
                str(inventory["model_class"]),
                str(inventory.get("model_type", "unknown")),
                inventory.get("source_revision"),
                str(inventory["status"]),
                canonical_json(compact),
            ),
        )
        modules = inventory.get("module_rows", [])
        for row in sorted(modules, key=lambda item: (item["canonical_path"].count("."), item["canonical_path"])):
            connection.execute(
                "INSERT INTO modules VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (snapshot_id, row["module_id"], row["canonical_path"], canonical_json(row["aliases"]), row.get("parent_module_id"), row["module_type"], canonical_json(row["signature"]), int(row["child_count"]), int(bool(row["is_leaf"]))),
            )
        for row in inventory.get("tensor_rows", []):
            connection.execute(
                "INSERT INTO tensors VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (snapshot_id, row["tensor_id"], row["state_name"], canonical_json(row["shape"]), row["dtype"], row.get("layout"), canonical_json(row["strides"]), row.get("element_count"), row.get("value_sha256"), row["hash_status"], row.get("storage_alias_group"), row["state_kind"]),
            )
        for row in inventory.get("bindings", []):
            connection.execute(
                "INSERT INTO module_tensor_bindings VALUES (?, ?, ?, ?, ?)",
                (snapshot_id, row["module_id"], row["tensor_id"], row["local_name"], row["relation"]),
            )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()
    return snapshot_id


def _module_records(database_path: str | Path, snapshot_id: str) -> list[dict[str, Any]]:
    connection = _connect(database_path)
    try:
        rows = connection.execute("SELECT * FROM modules WHERE snapshot_id = ? ORDER BY canonical_path", (snapshot_id,)).fetchall()
        return [dict(row) | {"aliases": json.loads(row["aliases_json"]), "signature": json.loads(row["signature_json"])} for row in rows]
    finally:
        connection.close()


def _insert_role_assertions(connection: sqlite3.Connection, snapshot_id: str, assertions: Iterable[Mapping[str, Any]]) -> None:
    for assertion in assertions:
        assertion_id = _digest("role-assertion", {
            "snapshot_id": snapshot_id,
            "target_kind": assertion["target_kind"],
            "target_id": assertion["target_id"],
            "taxonomy": assertion["taxonomy"],
            "role": assertion["role"],
            "status": assertion["status"],
            "evidence": assertion["evidence"],
            "detector_version": assertion["detector_version"],
        })
        connection.execute(
            "INSERT OR IGNORE INTO role_assertions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (assertion_id, snapshot_id, assertion["target_kind"], assertion["target_id"], assertion["taxonomy"], assertion["role"], assertion["status"], assertion.get("confidence"), assertion["detector_name"], assertion["detector_version"], canonical_json(assertion["evidence"]), _now()),
        )


def infer_structural_roles(
    inventory: Mapping[str, Any],
    graph: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Emit low-authority role candidates from registered structure and graph use.

    This intentionally uses no model-family table. Names/types are weak clues;
    observed graph calls add evidence that a module participates in execution.
    Every emitted claim stays a candidate, never a permission to mutate.
    """
    graph_nodes = (graph or {}).get("nodes", [])
    graph_calls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    module_by_id = {str(row["module_id"]): row for row in inventory.get("module_rows", [])}
    node_by_id = {str(node.get("node_id")): node for node in graph_nodes if node.get("node_id")}
    incoming: dict[str, list[str]] = defaultdict(list)
    for edge in (graph or {}).get("edges", []):
        incoming[str(edge.get("target_node_id"))].append(str(edge.get("source_node_id")))
    for node in graph_nodes:
        module_id = node.get("module_id")
        if module_id:
            graph_calls[str(module_id)].append(dict(node))
    role_patterns = (
        ("embedding_lookup", re.compile(r"embedding", re.IGNORECASE)),
        ("normalization", re.compile(r"norm(?:alization)?", re.IGNORECASE)),
        ("attention_component", re.compile(r"attention|self_attn|attn", re.IGNORECASE)),
        ("feed_forward_component", re.compile(r"feed.?forward|mlp|ffn", re.IGNORECASE)),
        ("routing_component", re.compile(r"router|routing|experts?.*gate|gate", re.IGNORECASE)),
        ("recurrent_or_stateful_component", re.compile(r"recurrent|state.?space|ssm|retention|rwkv|mixer", re.IGNORECASE)),
        ("linear_projection", re.compile(r"(?:^|\.)linear(?:$|\.)|linear$|projection", re.IGNORECASE)),
        ("repeated_block_candidate", re.compile(r"(?:^|\.)(?:layers?|blocks?|h)\.?(?:\d+)?$", re.IGNORECASE)),
    )
    assertions = []
    for module in inventory.get("module_rows", []):
        if not module.get("canonical_path"):
            continue
        text = f"{module.get('canonical_path', '')} {module.get('module_type', '')}"
        calls = graph_calls.get(module["module_id"], [])
        evidence_base = {
            "module_path": module.get("canonical_path"),
            "module_type": module.get("module_type"),
            "graph_call_count": len(calls),
            "graph_node_ids": [node.get("node_id") for node in calls],
            "source": ["registered_module_inventory"] + (["observed_execution_graph"] if calls else []),
        }
        for role, pattern in role_patterns:
            if pattern.search(text):
                assertions.append({
                    "target_kind": "module",
                    "target_id": module["module_id"],
                    "taxonomy": TAXONOMY_VERSION,
                    "role": role,
                    "status": "candidate",
                    "confidence": 0.5 if calls else 0.25,
                    "detector_name": _DETECTOR_NAME,
                    "detector_version": _DETECTOR_VERSION,
                    "evidence": evidence_base,
                })

    # Evidence-only attention pattern: a recognized attention operation has
    # dataflow ancestors from at least three distinct module-call projections.
    # Names are used only to locate the common registered parent, not to choose
    # a model family or assert a production-ready mapping.
    for node in graph_nodes:
        target = str(node.get("target", "")).lower()
        if not any(token in target for token in ("scaled_dot_product_attention", "multi_head_attention_forward")):
            continue
        ancestors: set[str] = set()
        pending = list(incoming.get(str(node.get("node_id")), []))
        visited: set[str] = set()
        while pending and len(visited) < 100_000:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            upstream = node_by_id.get(current)
            if upstream is None:
                continue
            if upstream.get("module_id"):
                ancestors.add(str(upstream["module_id"]))
            pending.extend(incoming.get(current, []))
        if len(ancestors) < 3:
            continue
        paths = [str(module_by_id[module_id]["canonical_path"]) for module_id in sorted(ancestors) if module_id in module_by_id]
        if len(paths) < 3:
            continue
        shared_segments = []
        for segments in zip(*(path.split(".") for path in paths)):
            if len(set(segments)) != 1:
                break
            shared_segments.append(segments[0])
        common_path = ".".join(shared_segments)
        parent = next((row for row in inventory.get("module_rows", []) if row.get("canonical_path") == common_path), None)
        if parent is None:
            continue
        assertions.append({
            "target_kind": "module",
            "target_id": parent["module_id"],
            "taxonomy": TAXONOMY_VERSION,
            "role": "attention_computation_candidate",
            "status": "candidate",
            "confidence": 0.65,
            "detector_name": _DETECTOR_NAME,
            "detector_version": _DETECTOR_VERSION,
            "evidence": {
                "graph_operator_node_id": node.get("node_id"),
                "graph_operator_target": node.get("target"),
                "upstream_module_ids": sorted(ancestors),
                "upstream_module_paths": paths,
                "common_parent_path": common_path,
                "source": ["observed_graph_dataflow", "multi_projection_attention_operator_pattern"],
            },
        })
    return assertions


def persist_role_assertions(database_path: str | Path, snapshot_id: str, assertions: Iterable[Mapping[str, Any]]) -> int:
    initialize_database(database_path)
    assertions = list(assertions)
    connection = _connect(database_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        before = connection.total_changes
        _insert_role_assertions(connection, snapshot_id, assertions)
        inserted = connection.total_changes - before
        connection.commit()
        return inserted
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def record_graph_observation(
    database_path: str | Path,
    snapshot_id: str,
    graph: Mapping[str, Any],
) -> str:
    """Append one graph trace; node attributes must not contain input values."""
    initialize_database(database_path)
    trace_id = str(graph.get("trace_id") or _digest("trace", {"snapshot_id": snapshot_id, "graph": graph}))
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))
    connection = _connect(database_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT INTO trace_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (trace_id, snapshot_id, str(graph.get("trace_kind", "unknown")), str(graph.get("status", "partial")), str(graph.get("created_at", _now())), canonical_json(_safe_json(graph.get("environment", {}))), canonical_json(_safe_json(graph.get("input_profile", {}))), canonical_json(_safe_json(graph.get("limitations", [])))),
        )
        for index, node in enumerate(nodes):
            node_id = str(node.get("node_id") or _digest("graph-node", {"trace_id": trace_id, "index": index, "node": node}))
            node["node_id"] = node_id
            connection.execute(
                "INSERT INTO graph_nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
                (trace_id, node_id, int(node.get("sequence", index)), str(node.get("node_kind", "unknown")), str(node.get("target", "unknown")), node.get("module_id"), canonical_json(_safe_json(node.get("attributes", {})))),
            )
        for edge in edges:
            connection.execute(
                "INSERT INTO graph_edges VALUES (?, ?, ?, ?, ?)",
                (trace_id, str(edge["source_node_id"]), str(edge["target_node_id"]), str(edge.get("relation", "data_flow")), canonical_json(_safe_json(edge.get("evidence", {})))),
            )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()
    return trace_id


def _shape_summary(value: Any) -> Any:
    shape = _shape(value)
    if shape is not None:
        return {"shape": shape, "dtype": str(getattr(value, "dtype", "unknown")), "device": str(getattr(value, "device", "unknown"))}
    if isinstance(value, Mapping):
        return {str(key): _shape_summary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_shape_summary(item) for item in value]
    return {"type": type(value).__name__}


def capture_execution_graph(
    isolated_model: Any,
    invoke: Callable[[Any], Any],
    *,
    inventory: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    input_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Trace actual nested module execution on a caller-provided isolated model.

    The capture stores call nesting and tensor metadata only. The caller must
    provide a disposable model copy because forwards can mutate buffers/caches.
    """
    if not callable(invoke):
        raise TypeError("invoke must be a callable accepting the isolated model.")
    state_before = _state_fingerprint(isolated_model)
    state_snapshot = copy.deepcopy(isolated_model.state_dict()) if callable(getattr(isolated_model, "state_dict", None)) else None
    capture_started = time.monotonic()
    named = _enumerate_named(isolated_model, "named_modules")
    if not named:
        return {"trace_kind": "module_execution", "status": "unsupported", "nodes": [], "edges": [], "limitations": ["named_modules_unavailable"]}
    inventory_ids = {row["canonical_path"]: row["module_id"] for row in (inventory or {}).get("module_rows", [])}
    runtime_to_info: dict[int, tuple[str, str, str | None]] = {}
    for path, module in named:
        runtime_to_info[id(module)] = (str(path), _type_name(module), inventory_ids.get(str(path)))
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    active: list[str] = []
    handles = []
    sequence = 0

    def pre_hook(module: Any, args: Any, kwargs: Any = None) -> None:
        nonlocal sequence
        path, module_type, module_id = runtime_to_info.get(id(module), ("", _type_name(module), None))
        node_id = f"call-{sequence:08d}"
        parent_id = active[-1] if active else None
        row = {
            "node_id": node_id,
            "sequence": sequence,
            "node_kind": "module_call",
            "target": path or module_type,
            "module_id": module_id,
            "attributes": {
                "module_path": path,
                "module_type": module_type,
                "input_signature": _shape_summary({"args": args, "kwargs": kwargs or {}}),
                "observed": True,
            },
        }
        sequence += 1
        nodes.append(row)
        if parent_id is not None:
            edges.append({"source_node_id": parent_id, "target_node_id": node_id, "relation": "module_call_parent", "evidence": {"observed_during_forward": True}})
        active.append(node_id)

    def post_hook(module: Any, args: Any, output: Any) -> None:
        path, module_type, _ = runtime_to_info.get(id(module), ("", _type_name(module), None))
        node = next((item for item in reversed(nodes) if item["target"] == (path or module_type) and item.get("attributes", {}).get("output_signature") is None and item["node_id"] in active), None)
        if node is not None:
            node["attributes"]["output_signature"] = _shape_summary(output)
            node["attributes"]["input_signature"] = node["attributes"].get("input_signature")
            try:
                active.remove(node["node_id"])
            except ValueError:
                pass

    try:
        for _, module in named:
            register_pre = getattr(module, "register_forward_pre_hook", None)
            register_post = getattr(module, "register_forward_hook", None)
            if not callable(register_pre) or not callable(register_post):
                continue
            try:
                handles.append(register_pre(pre_hook, with_kwargs=True))
            except TypeError:
                handles.append(register_pre(lambda mod, args, _hook=pre_hook: _hook(mod, args, {})))
            handles.append(register_post(post_hook))
        invoke(isolated_model)
        if _state_fingerprint(isolated_model) != state_before:
            loader = getattr(isolated_model, "load_state_dict", None)
            if state_snapshot is not None and callable(loader):
                loader(state_snapshot, strict=True)
            status = "failed"
            limitations = ["state_dict_mutated_during_trace"]
        else:
            status = "measured" if nodes else "partial"
            limitations = [] if nodes else ["no_module_calls_observed"]
    except Exception as exc:  # noqa: BLE001 -- model forwards are third-party code; persist failure evidence
        status = "failed"
        limitations = [f"forward_failed:{type(exc).__name__}"]
    finally:
        for handle in handles:
            try:
                handle.remove()
            except (AttributeError, RuntimeError):
                pass
    return {
        "trace_id": _digest("trace", {"kind": "module_execution", "nodes": nodes, "input_profile": _safe_json(input_profile or {}), "created_at": _now()}),
        "trace_kind": "module_execution",
        "status": status,
        "created_at": _now(),
        "environment": _safe_json(environment or {}),
        "input_profile": _safe_json(input_profile or {}),
        "nodes": nodes,
        "edges": edges,
        "limitations": limitations + ["mutation_guard_covers_state_dict_only; use a disposable copy for runtime caches and external side effects"],
        "capture_duration_seconds": time.monotonic() - capture_started,
    }


def capture_fx_graph(
    isolated_model: Any,
    *,
    inventory: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Best-effort PyTorch FX graph capture; unsupported models stay explicit."""
    capture_started = time.monotonic()
    try:
        import torch.fx
    except ImportError:
        return {"trace_kind": "torch_fx", "status": "unsupported", "nodes": [], "edges": [], "limitations": ["pytorch_fx_unavailable"]}
    try:
        traced = torch.fx.symbolic_trace(isolated_model)
        nodes = list(traced.graph.nodes)
    except Exception as exc:  # noqa: BLE001 -- arbitrary FX tracing failures are recorded as unsupported
        return {"trace_kind": "torch_fx", "status": "unsupported", "nodes": [], "edges": [], "limitations": [f"symbolic_trace_failed:{type(exc).__name__}"]}
    module_ids = {row["canonical_path"]: row["module_id"] for row in (inventory or {}).get("module_rows", [])}
    result_nodes = []
    result_edges = []
    node_ids = {node: _digest("fx-node", {"name": node.name, "op": node.op, "target": str(node.target)}) for node in nodes}
    for index, node in enumerate(nodes):
        target = str(node.target)
        module_id = module_ids.get(target) if node.op == "call_module" else None
        result_nodes.append({
            "node_id": node_ids[node],
            "sequence": index,
            "node_kind": str(node.op),
            "target": target,
            "module_id": module_id,
            "attributes": {"name": str(node.name), "args_repr": str(node.args)[:2000], "kwargs_repr": str(node.kwargs)[:2000]},
        })
        dependencies = [item for item in node.all_input_nodes]
        for dependency in dependencies:
            result_edges.append({"source_node_id": node_ids[dependency], "target_node_id": node_ids[node], "relation": "data_flow", "evidence": {"source": "torch.fx.Node.all_input_nodes"}})
    return {
        "trace_id": _digest("trace", {"kind": "torch_fx", "nodes": result_nodes, "edges": result_edges, "created_at": _now()}),
        "trace_kind": "torch_fx",
        "status": "measured",
        "created_at": _now(),
        "environment": _safe_json(environment or {}),
        "input_profile": {},
        "nodes": result_nodes,
        "edges": result_edges,
        "limitations": ["symbolic_graph_only; validate against concrete execution traces"],
        "capture_duration_seconds": time.monotonic() - capture_started,
    }


def _path_get(mapping: Mapping[str, Any], dotted: str) -> Any:
    current: Any = mapping
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _valid_profile_measurement(profile: str, measurement: Mapping[str, Any]) -> tuple[bool, str | None]:
    paths: dict[str, tuple[str, ...]] = {
        "output_sensitivity": ("baseline.output_signature", "ablated.output_signature", "delta.output_distance"),
        "capability_axes": ("baseline.capabilities", "ablated.capabilities"),
        "input_regime_robustness": ("regimes",),
        "resource_impact": ("baseline.latency_ms", "ablated.latency_ms", "baseline.peak_memory_bytes", "ablated.peak_memory_bytes"),
        "stability_recovery": (
            "stability.baseline", "stability.after_removal", "stability.after_recovery",
            "stability.recovery_method", "stability.recovery_steps", "stability.balance_score",
            "stability.balance_criterion", "stability.balance_axes",
        ),
    }
    required = paths.get(profile)
    if required is None:
        return False, "unknown_probe_profile"
    missing = [path for path in required if _path_get(measurement, path) is None]
    if missing:
        return False, "missing_required_metrics:" + ",".join(missing)
    if profile == "capability_axes":
        before, after = _path_get(measurement, "baseline.capabilities"), _path_get(measurement, "ablated.capabilities")
        if not isinstance(before, Mapping) or not isinstance(after, Mapping) or not set(_REQUIRED_AXES).issubset(before) or not set(_REQUIRED_AXES).issubset(after):
            return False, "required_capability_axes_missing"
        if any(not _finite_number(values[axis]) for values in (before, after) for axis in _REQUIRED_AXES):
            return False, "capability_axes_must_be_finite_numbers"
    elif profile == "output_sensitivity" and not _finite_number(_path_get(measurement, "delta.output_distance")):
        return False, "output_distance_must_be_finite_number"
    elif profile == "input_regime_robustness":
        regimes = measurement.get("regimes")
        if not isinstance(regimes, Mapping) or len(regimes) < 2 or any(not isinstance(value, Mapping) or not _finite_number(value.get("baseline")) or not _finite_number(value.get("ablated")) for value in regimes.values()):
            return False, "at_least_two_baseline_and_ablated_regimes_required"
    elif profile == "resource_impact":
        metric_paths = ("baseline.latency_ms", "ablated.latency_ms", "baseline.peak_memory_bytes", "ablated.peak_memory_bytes")
        if any(not _finite_number(_path_get(measurement, path)) for path in metric_paths):
            return False, "resource_metrics_must_be_finite_numbers"
    elif profile == "stability_recovery":
        stability = measurement.get("stability", {})
        axes = stability.get("balance_axes", {}) if isinstance(stability, Mapping) else {}
        if not isinstance(axes, Mapping) or not set(_REQUIRED_AXES).issubset(axes):
            return False, "balance_axes_reasoning_general_cohesion_required"
        numeric_fields = ("baseline", "after_removal", "after_recovery", "balance_score")
        if any(not _finite_number(stability.get(key)) for key in numeric_fields) or any(not _finite_number(axes[key]) for key in _REQUIRED_AXES):
            return False, "stability_and_balance_metrics_must_be_finite_numbers"
        if not isinstance(stability.get("recovery_method"), str) or not stability["recovery_method"]:
            return False, "recovery_method_required"
        if not isinstance(stability.get("balance_criterion"), str) or not stability["balance_criterion"]:
            return False, "balance_criterion_required"
        if isinstance(stability.get("recovery_steps"), bool) or not isinstance(stability.get("recovery_steps"), int) or stability["recovery_steps"] < 0:
            return False, "recovery_steps_must_be_nonnegative_integer"
    # Reject NaN/Infinity anywhere before it enters the immutable evidence log.
    try:
        canonical_json(measurement)
    except (TypeError, ValueError):
        return False, "measurement_not_finite_json"
    return True, None


def _state_fingerprint(model: Any) -> str:
    state = getattr(model, "state_dict", None)
    if not callable(state):
        raise TypeError("Cannot run guarded tracing/probes without a state_dict API.")
    try:
        values = state()
    except Exception as exc:
        raise ValueError("Cannot fingerprint model state for guarded tracing/probes.") from exc
    if not isinstance(values, Mapping):
        raise TypeError("state_dict must return a mapping for guarded tracing/probes.")
    rows = []
    for name, value in sorted(values.items(), key=lambda item: str(item[0])):
        metadata = _tensor_metadata(value)
        raw = _tensor_bytes(value)
        if raw is None:
            raise ValueError("Cannot fingerprint a state_dict value; guarded tracing/probes are disabled.")
        rows.append((str(name), metadata, hashlib.sha256(raw).hexdigest()))
    return _digest("state", rows)


def _strip_untrusted_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _strip_untrusted_fields(item) for key, item in value.items() if str(key).lower() not in {"label", "user_label", "promotion", "permission", "authorized"}}
    if isinstance(value, (list, tuple)):
        return [_strip_untrusted_fields(item) for item in value]
    return _safe_json(value)


def _centrality_warnings(probes: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    capability = next((item for item in probes if item["probe_name"] == "capability_axes" and item["status"] == "measured"), None)
    if capability is None:
        return [{"code": "centrality_unmeasured", "message": "Reasoning, general capability, and cohesion impacts were not fully measured; do not treat this layer as low-risk."}]
    before = capability["measurement"].get("baseline", {}).get("capabilities", {})
    after = capability["measurement"].get("ablated", {}).get("capabilities", {})
    warnings = []
    for axis in _REQUIRED_AXES:
        try:
            delta = float(after[axis]) - float(before[axis])
        except (KeyError, TypeError, ValueError):
            warnings.append({"code": "centrality_axis_unmeasured", "axis": axis, "message": f"No valid {axis} baseline/ablation comparison was recorded."})
            continue
        if abs(delta) >= threshold:
            warnings.append({"code": "high_dependency_risk", "axis": axis, "observed_delta": delta, "threshold": threshold, "message": f"Ablation changed the measured {axis} capability by {delta:.6g}; removal may affect this capability."})
    return warnings


def run_layer_probe_battery(
    model: Any,
    inventory: Mapping[str, Any],
    database_path: str | Path,
    adapter: Any,
    *,
    profiles: Iterable[str] = DEFAULT_PROBE_PROFILES,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run multiple isolated causal probes on every inventoried non-root module.

    ``adapter.sandbox_layer_probe`` must create/use disposable model copies,
    remove or disable only the requested module, measure outcomes, and return
    ``{"isolated_copy": true, "measurements": {...}}``. The live model is
    fingerprint-checked and best-effort restored if an adapter violates the
    contract. No result grants mutation authority.
    """
    profiles = tuple(dict.fromkeys(str(profile) for profile in profiles))
    if set(profiles) != set(DEFAULT_PROBE_PROFILES):
        raise ValueError("Every layer assessment must run the complete five-profile probe suite.")
    unknown_profiles = set(profiles) - set(DEFAULT_PROBE_PROFILES)
    if unknown_profiles:
        raise ValueError(f"Unsupported probe profiles: {', '.join(sorted(unknown_profiles))}")
    max_modules = int((context or {}).get("inventory_max_modules", 100_000))
    max_seconds = float((context or {}).get("inventory_max_seconds", 86_400.0))
    if max_modules < 1 or max_modules > 100_000 or not math.isfinite(max_seconds) or not 0 < max_seconds <= 86_400:
        raise ValueError("Inventory probe bounds must be positive and within 100000 modules/86400 seconds.")
    probe = getattr(adapter, "sandbox_layer_probe", None)
    module_rows = [row for row in inventory.get("module_rows", []) if row.get("canonical_path")]
    module_rows.sort(key=lambda row: row["canonical_path"])
    started = time.monotonic()
    started_at = _now()
    baseline_live = _state_fingerprint(model)
    probe_records: list[dict[str, Any]] = []
    truncated = False
    threshold = float((context or {}).get("centrality_warning_delta_threshold", 0.05))
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("centrality_warning_delta_threshold must be finite and nonnegative.")
    for module_index, module in enumerate(module_rows):
        if module_index >= max_modules or time.monotonic() - started > max_seconds:
            truncated = True
            break
        module_probes: list[dict[str, Any]] = []
        for profile in profiles:
            if time.monotonic() - started > max_seconds:
                truncated = True
                break
            probe_id = str(uuid.uuid4())
            result = {
                "probe_id": probe_id,
                "module_id": module["module_id"],
                "module_path": module["canonical_path"],
                "probe_name": profile,
                "status": "unknown",
                "isolated": False,
                "measurement": {},
                "warnings": [],
                "limitations": [],
            }
            before = _state_fingerprint(model)
            state_snapshot = None
            state_method = getattr(model, "state_dict", None)
            loader = getattr(model, "load_state_dict", None)
            try:
                if callable(state_method):
                    state_snapshot = copy.deepcopy(state_method())
                if not callable(probe):
                    result["limitations"].append("sandbox_layer_probe_unavailable")
                else:
                    raw = probe(dict(module), profile, dict(context or {}))
                    after = _state_fingerprint(model)
                    if before != after:
                        result["status"] = "failed"
                        result["limitations"].append("live_model_mutated_by_probe_adapter")
                        if state_snapshot is not None and callable(loader):
                            loader(state_snapshot, strict=True)
                        if _state_fingerprint(model) != baseline_live:
                            result["limitations"].append("live_model_restore_verification_failed")
                    elif not isinstance(raw, Mapping) or raw.get("isolated_copy") is not True:
                        result["limitations"].append("probe_did_not_attest_isolated_copy")
                    elif raw.get("removal_performed") is not True or raw.get("removed_module_id") != module["module_id"]:
                        result["limitations"].append("requested_module_removal_not_verified")
                    elif not isinstance(raw.get("measurements"), Mapping):
                        result["limitations"].append("probe_measurements_missing")
                    else:
                        clean = _strip_untrusted_fields(raw["measurements"])
                        valid, reason = _valid_profile_measurement(profile, clean)
                        result["isolated"] = True
                        result["measurement"] = clean
                        result["status"] = "measured" if valid else "unknown"
                        if not valid and reason:
                            result["limitations"].append(reason)
                        if valid:
                            result["measurement_signature"] = _digest("measurement", clean)
            except Exception as exc:  # noqa: BLE001 -- probe adapter failures are durable evidence
                if state_snapshot is not None and callable(loader):
                    try:
                        loader(state_snapshot, strict=True)
                    except Exception:  # noqa: BLE001 -- restoration failure must be reported, not hidden
                        result["limitations"].append("live_model_restore_failed")
                result["status"] = "failed"
                result["limitations"].append(f"probe_exception:{type(exc).__name__}")
            module_probes.append(result)
            probe_records.append(result)
        warnings = _centrality_warnings(module_probes, threshold)
        for result in module_probes:
            result["warnings"] = warnings if result["probe_name"] in {"capability_axes", "stability_recovery"} else []
        if truncated:
            break
    final_live = _state_fingerprint(model)
    if final_live != baseline_live:
        truncated = True
        root_module = next((row for row in inventory.get("module_rows", []) if not row.get("canonical_path")), None)
        guard_module_id = root_module["module_id"] if root_module else (module_rows[0]["module_id"] if module_rows else "")
        probe_records.append({
            "probe_id": str(uuid.uuid4()), "module_id": guard_module_id, "module_path": "", "probe_name": "live_state_guard",
            "status": "failed", "isolated": False, "measurement": {}, "warnings": [],
            "limitations": ["live_model_state_changed_during_probe_battery"],
        })
    finished_at = _now()
    measured = sum(item["status"] == "measured" for item in probe_records)
    unknown = sum(item["status"] == "unknown" for item in probe_records)
    failed = sum(item["status"] == "failed" for item in probe_records)
    status = "failed" if failed else "partial" if truncated or unknown else "measured"
    report = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": inventory["snapshot_id"],
        "probe_run_id": _digest("probe-run", {"snapshot_id": inventory["snapshot_id"], "started_at": started_at, "profiles": profiles}),
        "status": status,
        "target_module_count": len(module_rows),
        "attempted_probe_count": len(probe_records),
        "measured_probe_count": measured,
        "unknown_probe_count": unknown,
        "failed_probe_count": failed,
        "profiles": list(profiles),
        "probe_adapter": {
            "type": _type_name(adapter),
            "name": str(getattr(adapter, "inventory_probe_name", _type_name(adapter))),
            "version": str(getattr(adapter, "inventory_probe_version", "unversioned")),
        },
        "probe_suite_id": str((context or {}).get("inventory_probe_suite_id", "unversioned")),
        "probe_dataset_id": str((context or {}).get("inventory_probe_dataset_id", "unversioned")),
        "probes": probe_records,
        "layer_warnings": {
            module["module_id"]: _centrality_warnings([item for item in probe_records if item.get("module_id") == module["module_id"]], threshold)
            for module in module_rows
        },
        "model_modified": final_live != baseline_live,
        "transient_live_mutation_detected": any("live_model_mutated_by_probe_adapter" in item.get("limitations", []) for item in probe_records),
        "runtime_permission_granted": False,
        "truncated": truncated,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": max(0.0, time.monotonic() - started),
        "probes_per_second": len(probe_records) / max(time.monotonic() - started, 1e-9),
        "limitations": ["Results describe only the supplied probe suite and test data.", "No universal definition of model stability or balanced capability exists; the adapter must record its recovery method and balance criterion.", "Removal warnings report observed risk and do not block or authorize edits."],
    }
    persist_probe_report(database_path, report)
    return report


def persist_probe_report(database_path: str | Path, report: Mapping[str, Any]) -> None:
    initialize_database(database_path)
    connection = _connect(database_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT INTO probe_runs VALUES (?, ?, ?, ?, ?, ?)",
            (report["probe_run_id"], report["snapshot_id"], report["status"], report["started_at"], report["finished_at"], canonical_json({key: report.get(key) for key in ("target_module_count", "attempted_probe_count", "measured_probe_count", "unknown_probe_count", "failed_probe_count", "profiles", "probe_adapter", "probe_suite_id", "probe_dataset_id", "truncated", "elapsed_seconds", "probes_per_second", "limitations", "layer_warnings")})),
        )
        for probe in report.get("probes", []):
            connection.execute(
                "INSERT INTO layer_probes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (probe["probe_id"], report["probe_run_id"], report["snapshot_id"], probe["module_id"], probe["module_path"], probe["probe_name"], probe["status"], int(bool(probe["isolated"])), probe.get("measurement_signature"), canonical_json(probe["measurement"]), canonical_json(probe["warnings"]), canonical_json(probe["limitations"]), _now()),
            )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _bind_probe_evidence(
    assertions: Iterable[Mapping[str, Any]],
    probe_report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_module: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for probe in probe_report.get("probes", []):
        by_module[str(probe.get("module_id", ""))].append(probe)
    enriched = []
    for assertion in assertions:
        row = dict(assertion)
        evidence = dict(row.get("evidence", {}))
        probes = by_module.get(str(row.get("target_id", "")), [])
        measured = [probe for probe in probes if probe.get("status") == "measured"]
        evidence["probe_evidence"] = {
            "probe_run_id": probe_report.get("probe_run_id"),
            "probe_ids": [probe.get("probe_id") for probe in probes],
            "profiles_attempted": [probe.get("probe_name") for probe in probes],
            "profiles_measured": [probe.get("probe_name") for probe in measured],
            "profiles_complete": set(probe_report.get("profiles", [])) == {probe.get("probe_name") for probe in measured},
            "warning_codes": sorted({warning.get("code") for probe in probes for warning in probe.get("warnings", []) if warning.get("code")}),
            "measurement_signatures": {probe["probe_name"]: probe.get("measurement_signature") for probe in measured},
            "semantic_role_not_promoted": True,
        }
        row["evidence"] = evidence
        # Role evidence is attached for auditability, but causal impact does
        # not establish semantic identity; candidate status is retained.
        row["status"] = "candidate"
        enriched.append(row)
    return enriched


def evaluate_unseen_architectures(
    gold: Iterable[Mapping[str, Any]],
    predicted: Iterable[Mapping[str, Any]],
    *,
    dataset_id: str,
    universe_targets: Iterable[Mapping[str, Any]],
    held_out_group_ids: Iterable[str],
    training_group_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Score role labels on architecture groups absent from the training set.

    Rows require ``group_id``, ``target_id``, and ``role``. This computes binary
    per-role precision/recall/F1, coverage, and false-positive rate. It does not
    provide a benchmark corpus; callers must supply independently annotated,
    held-out architecture evidence.
    """
    held_out = set(map(str, held_out_group_ids))
    training = set(map(str, training_group_ids))
    overlap = held_out & training
    if not held_out or overlap:
        raise ValueError("Held-out group IDs must be non-empty and disjoint from training group IDs.")
    gold_rows = list(gold)
    pred_rows = list(predicted)
    universe_rows = list(universe_targets)
    universe = {(str(row["group_id"]), str(row["target_id"])) for row in universe_rows if str(row["group_id"]) in held_out}
    if not universe:
        raise ValueError("The complete held-out target universe is required to calculate false-positive rates.")
    def keys(rows: list[Mapping[str, Any]], label: str) -> set[tuple[str, str, str]]:
        result = set()
        for row in rows:
            group = str(row["group_id"])
            if group not in held_out:
                continue
            result.add((group, str(row["target_id"]), str(row["role"])))
        return result
    gold_set = keys(gold_rows, "gold")
    pred_set = keys(pred_rows, "predicted")
    universe_mismatch = {(group, target) for group, target, _ in gold_set | pred_set} - universe
    if universe_mismatch:
        raise ValueError("Gold/predicted labels contain targets outside the declared held-out universe.")
    roles = sorted({item[2] for item in gold_set | pred_set})
    metrics = {}
    for role in roles:
        true_items = {item for item in gold_set if item[2] == role}
        pred_items = {item for item in pred_set if item[2] == role}
        tp = len(true_items & pred_items)
        fp = len(pred_items - true_items)
        fn = len(true_items - pred_items)
        positives = {(group, target) for group, target, _ in true_items}
        tn = len(universe - positives - pred_items)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[role] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1, "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0}
    coverage = len({(group, target) for group, target, _ in pred_set}) / len(universe)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "split_name": "unseen_architecture_holdout",
        "held_out_group_ids": sorted(held_out),
        "metrics": {"per_role": metrics, "gold_label_count": len(gold_set), "prediction_count": len(pred_set), "held_out_target_count": len(universe), "target_coverage": coverage},
        "warning": "Evaluation quality depends on the independently annotated gold set; synthetic fixtures are not evidence of generalization to real model families.",
    }


def persist_evaluation(database_path: str | Path, evaluation: Mapping[str, Any], *, detector_name: str = _DETECTOR_NAME, detector_version: str = _DETECTOR_VERSION) -> str:
    initialize_database(database_path)
    evaluation_id = _digest("evaluation", {"dataset_id": evaluation["dataset_id"], "held_out_group_ids": evaluation["held_out_group_ids"], "metrics": evaluation["metrics"], "detector": [detector_name, detector_version]})
    connection = _connect(database_path)
    try:
        connection.execute("INSERT OR IGNORE INTO evaluation_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (evaluation_id, evaluation["dataset_id"], evaluation["split_name"], detector_name, detector_version, canonical_json(evaluation["metrics"]), canonical_json(evaluation["held_out_group_ids"]), _now()))
        connection.commit()
    finally:
        connection.close()
    return evaluation_id


def build_model_inventory(
    model: Any,
    database_path: str | Path,
    *,
    source_revision: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    isolated_model: Any | None = None,
    invoke: Callable[[Any], Any] | None = None,
    probe_adapter: Any | None = None,
    probe_context: Mapping[str, Any] | None = None,
    capture_fx: bool = True,
) -> dict[str, Any]:
    """Build and persist a snapshot, graph evidence, candidate roles, and probes.

    Graph tracing must use a distinct disposable model instance with an exact
    state/config match. Isolated removal tests run only when an explicit probe
    adapter is supplied. The returned report never grants mutation authority.
    """
    inventory = capture_inventory(model, source_revision=source_revision, metadata=metadata)
    if isolated_model is model:
        raise PermissionError("Graph capture requires a separate disposable model copy.")
    isolated_inventory = None
    if isolated_model is not None:
        isolated_inventory = capture_inventory(isolated_model, source_revision=source_revision, metadata=metadata)
        if isolated_inventory["snapshot_id"] != inventory["snapshot_id"]:
            raise ValueError("Disposable graph-capture model does not match the inventoried snapshot.")
    persist_inventory(database_path, inventory)
    graphs: list[dict[str, Any]] = []
    if isolated_model is not None:
        if invoke is not None:
            graph = capture_execution_graph(
                isolated_model,
                invoke,
                inventory=inventory,
                environment=metadata,
            )
            record_graph_observation(database_path, inventory["snapshot_id"], graph)
            graphs.append(graph)
        if capture_fx:
            graph = capture_fx_graph(isolated_model, inventory=inventory, environment=metadata)
            record_graph_observation(database_path, inventory["snapshot_id"], graph)
            graphs.append(graph)
    elif invoke is not None or capture_fx:
        graphs.append({
            "trace_kind": "not_run",
            "status": "unsupported",
            "nodes": [],
            "edges": [],
            "limitations": ["pass a separate isolated_model to enable graph capture"],
        })
    assertions = infer_structural_roles(inventory)
    for graph in graphs:
        assertions.extend(infer_structural_roles(inventory, graph))
    # Deterministically deduplicate role candidates while preserving evidence
    # from separate capture modes as separate assertions when it differs.
    unique_assertions = {
        _digest("role-candidate", {
            "target_id": item["target_id"],
            "role": item["role"],
            "evidence": item["evidence"],
        }): item
        for item in assertions
    }
    role_assertions = list(unique_assertions.values())
    probes = None
    if probe_adapter is not None:
        probes = run_layer_probe_battery(
            model,
            inventory,
            database_path,
            probe_adapter,
            context=probe_context,
        )
        role_assertions = _bind_probe_evidence(role_assertions, probes)
    persist_role_assertions(database_path, inventory["snapshot_id"], role_assertions)
    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": inventory["snapshot_id"],
        "structure_id": inventory["structure_id"],
        "inventory_status": inventory["status"],
        "module_count": inventory["module_count"],
        "tensor_count": inventory["tensor_count"],
        "capture_duration_seconds": inventory["capture_duration_seconds"],
        "graphs": graphs,
        "role_assertions": role_assertions,
        "probe_report": probes,
        "model_modified": False,
        "runtime_permission_granted": False,
        "warnings": [
            "Role assertions are hypotheses supported by observed structure/calls; they never authorize mutation.",
            "Graph capture and isolated probes require adapter/toolchain support and representative inputs.",
        ],
    }


def inventory_summary(database_path: str | Path, snapshot_id: str) -> dict[str, Any]:
    connection = _connect(database_path)
    try:
        snapshot = connection.execute("SELECT * FROM inventory_snapshots WHERE snapshot_id = ?", (snapshot_id,)).fetchone()
        if snapshot is None:
            raise KeyError(f"Unknown model inventory snapshot: {snapshot_id}")
        counts = {}
        for table in ("modules", "tensors", "trace_runs", "role_assertions", "layer_probes"):
            counts[table] = int(connection.execute(f"SELECT COUNT(*) FROM {table} WHERE snapshot_id = ?", (snapshot_id,)).fetchone()[0])
        return {"snapshot_id": snapshot_id, "structure_id": snapshot["structure_id"], "status": snapshot["status"], **counts}
    finally:
        connection.close()
