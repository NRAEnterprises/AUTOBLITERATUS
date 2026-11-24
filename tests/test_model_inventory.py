from __future__ import annotations

import copy
import sqlite3
from types import SimpleNamespace

import pytest

from autobliteratus import discovery_pipeline as sandbox_discovery
from autobliteratus.core import discovery_experiment_pipeline as experiment_pipeline
from autobliteratus.core import model_mapping_discovery
from autobliteratus.core.model_inventory import (
    DEFAULT_PROBE_PROFILES,
    _state_fingerprint,
    build_model_inventory,
    capture_execution_graph,
    capture_inventory,
    evaluate_unseen_architectures,
    infer_structural_roles,
    initialize_database,
    inventory_summary,
    persist_inventory,
    record_graph_observation,
    run_layer_probe_battery,
)


class Handle:
    def __init__(self, remove):
        self._remove = remove

    def remove(self):
        self._remove()


class Weight:
    shape = (2, 2)
    dtype = "float32"
    layout = "strided"

    def __init__(self, raw=b"abcd"):
        self.raw = raw

    def stride(self):
        return (2, 1)

    def numel(self):
        return 4

    def tobytes(self, order="C"):
        del order
        return self.raw


class FakeModule:
    def __init__(self):
        self._children = {}
        self._pre_hooks = []
        self._post_hooks = []
        self.weight = Weight()

    def add_module(self, name, module):
        self._children[name] = module
        setattr(self, name, module)

    def named_children(self):
        return list(self._children.items())

    def named_modules(self, remove_duplicate=False):
        del remove_duplicate
        yield "", self
        for name, child in self._children.items():
            yield name, child
            for subname, submodule in child.named_modules():
                if subname:
                    yield f"{name}.{subname}", submodule

    def named_parameters(self, remove_duplicate=False):
        del remove_duplicate
        rows = []
        if hasattr(self, "weight"):
            rows.append(("weight", self.weight))
        for name, child in self._children.items():
            rows.extend((f"{name}.{key}", value) for key, value in child.named_parameters())
        return rows

    def named_buffers(self, remove_duplicate=False):
        del remove_duplicate
        return []

    def state_dict(self):
        return {name: value for name, value in self.named_parameters()}

    def register_forward_pre_hook(self, hook, with_kwargs=False):
        entry = (hook, with_kwargs)
        self._pre_hooks.append(entry)
        return Handle(lambda: self._pre_hooks.remove(entry))

    def register_forward_hook(self, hook):
        self._post_hooks.append(hook)
        return Handle(lambda: self._post_hooks.remove(hook))

    def __call__(self, value):
        args = (value,)
        for hook, with_kwargs in self._pre_hooks:
            if with_kwargs:
                hook(self, args, {})
            else:
                hook(self, args)
        result = self.forward(value)
        for hook in self._post_hooks:
            hook(self, args, result)
        return result

    def forward(self, value):
        for child in self._children.values():
            value = child(value)
        return value


class FakeAttention(FakeModule):
    pass


class FakeModel(FakeModule):
    def __init__(self):
        super().__init__()
        self.blocks = FakeModule()
        self.add_module("blocks", self.blocks)
        self.blocks.add_module("0", FakeModule())
        self.blocks._children["0"].add_module("attention", FakeAttention())
        self.blocks._children["0"]._children["attention"].weight = Weight(b"wxyz")
        self.config = {"model_type": "unseen_demo", "hidden_size": 2}

    def state_dict(self):
        return {
            "blocks.0.attention.weight": self.blocks._children["0"]._children["attention"].weight,
            "other.weight": Weight(b"lmno"),
        }

    def load_state_dict(self, state, strict=True):
        del strict
        self.blocks._children["0"]._children["attention"].weight.raw = state["blocks.0.attention.weight"].raw

    def forward(self, value):
        return self.blocks(value)


def _good_measurement(profile):
    if profile == "output_sensitivity":
        return {"baseline": {"output_signature": "base"}, "ablated": {"output_signature": "changed"}, "delta": {"output_distance": 0.2}}
    if profile == "capability_axes":
        return {
            "baseline": {"capabilities": {"reasoning": 0.9, "general": 0.8, "cohesion": 0.85}},
            "ablated": {"capabilities": {"reasoning": 0.5, "general": 0.7, "cohesion": 0.6}},
        }
    if profile == "input_regime_robustness":
        return {"regimes": {"short": {"baseline": 0.9, "ablated": 0.7}, "long": {"baseline": 0.8, "ablated": 0.5}}}
    if profile == "resource_impact":
        return {"baseline": {"latency_ms": 10.0, "peak_memory_bytes": 1000}, "ablated": {"latency_ms": 8.0, "peak_memory_bytes": 800}}
    if profile == "stability_recovery":
        return {
            "stability": {
                "baseline": 0.95,
                "after_removal": 0.3,
                "after_recovery": 0.8,
                "recovery_method": "adapter-defined balancing procedure",
                "recovery_steps": 3,
                "balance_score": 0.78,
                "balance_criterion": "task-axis minimum score >= 0.5",
                "balance_axes": {"reasoning": 0.6, "general": 0.7, "cohesion": 0.8},
            }
        }
    raise AssertionError(profile)


def test_inventory_ids_are_stable_and_sqlite_records_are_immutable(tmp_path):
    model = FakeModel()
    first = capture_inventory(model, source_revision="demo@1")
    second = capture_inventory(model, source_revision="demo@1")
    assert first["snapshot_id"] == second["snapshot_id"]
    assert first["structure_id"] == second["structure_id"]
    changed_weights = copy.deepcopy(model)
    changed_weights.blocks._children["0"]._children["attention"].weight.raw = b"diff"
    changed = capture_inventory(changed_weights, source_revision="demo@1")
    assert changed["structure_id"] == first["structure_id"]
    assert changed["snapshot_id"] != first["snapshot_id"]
    assert first["status"] == "complete"
    assert len(first["module_rows"]) >= 4
    assert first["tensor_rows"][0]["hash_status"] == "verified"

    database = tmp_path / "inventory.sqlite"
    assert persist_inventory(database, first) == first["snapshot_id"]
    assert persist_inventory(database, second) == first["snapshot_id"]
    summary = inventory_summary(database, first["snapshot_id"])
    assert summary["modules"] == len(first["module_rows"])
    assert summary["tensors"] == len(first["tensor_rows"])

    connection = sqlite3.connect(database)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        connection.execute("UPDATE modules SET module_type = 'tampered' WHERE snapshot_id = ?", (first["snapshot_id"],))
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        connection.execute("DELETE FROM tensors WHERE snapshot_id = ?", (first["snapshot_id"],))
    connection.close()


def test_graph_trace_and_role_candidates_use_observed_calls(tmp_path):
    model = FakeModel()
    inventory = capture_inventory(model)
    graph = capture_execution_graph(
        model,
        lambda candidate: candidate(1),
        inventory=inventory,
        input_profile={"sequence_length": 1},
    )
    assert graph["status"] == "measured"
    assert graph["nodes"]
    assert graph["edges"]
    assert "input_signature" in graph["nodes"][0]["attributes"]
    assert "output_signature" in graph["nodes"][0]["attributes"]

    database = tmp_path / "graph.sqlite"
    persist_inventory(database, inventory)
    trace_id = record_graph_observation(database, inventory["snapshot_id"], graph)
    assert trace_id
    assertions = infer_structural_roles(inventory, graph)
    attention_roles = [row for row in assertions if row["role"] == "attention_component"]
    assert attention_roles
    assert all(row["status"] == "candidate" for row in attention_roles)
    assert any(row["evidence"]["graph_call_count"] > 0 for row in attention_roles)


def test_attention_role_candidate_requires_graph_dataflow_from_three_modules():
    rows = [
        {"module_id": "parent", "canonical_path": "blocks.0.attention", "module_type": "custom.Container"},
        {"module_id": "q", "canonical_path": "blocks.0.attention.proj_a", "module_type": "custom.Linear"},
        {"module_id": "k", "canonical_path": "blocks.0.attention.proj_b", "module_type": "custom.Linear"},
        {"module_id": "v", "canonical_path": "blocks.0.attention.proj_c", "module_type": "custom.Linear"},
    ]
    graph = {
        "nodes": [
            {"node_id": "q-call", "node_kind": "call_module", "target": "proj_a", "module_id": "q"},
            {"node_id": "k-call", "node_kind": "call_module", "target": "proj_b", "module_id": "k"},
            {"node_id": "v-call", "node_kind": "call_module", "target": "proj_c", "module_id": "v"},
            {"node_id": "sdpa", "node_kind": "call_function", "target": "torch.nn.functional.scaled_dot_product_attention", "module_id": None},
        ],
        "edges": [
            {"source_node_id": source, "target_node_id": "sdpa", "relation": "data_flow"}
            for source in ("q-call", "k-call", "v-call")
        ],
    }
    roles = infer_structural_roles({"module_rows": rows}, graph)
    claim = next(row for row in roles if row["role"] == "attention_computation_candidate")
    assert claim["target_id"] == "parent"
    assert claim["status"] == "candidate"
    assert len(claim["evidence"]["upstream_module_ids"]) == 3


def test_each_module_gets_multiple_probe_profiles_and_centrality_warnings(tmp_path):
    model = FakeModel()
    inventory = capture_inventory(model)
    database = tmp_path / "probes.sqlite"
    persist_inventory(database, inventory)

    class ProbeAdapter:
        def __init__(self):
            self.calls = []

        def sandbox_layer_probe(self, module, profile, context):
            self.calls.append((module["module_id"], profile))
            del context
            return {
                "isolated_copy": True,
                "removal_performed": True,
                "removed_module_id": module["module_id"],
                "measurements": _good_measurement(profile),
                "user_label": "must not persist",
            }

    adapter = ProbeAdapter()
    report = run_layer_probe_battery(model, inventory, database, adapter)
    nonroot_modules = [row for row in inventory["module_rows"] if row["canonical_path"]]
    assert report["status"] == "measured"
    assert report["model_modified"] is False
    assert report["runtime_permission_granted"] is False
    assert report["attempted_probe_count"] == len(nonroot_modules) * len(DEFAULT_PROBE_PROFILES)
    assert all({profile for module_id, profile in adapter.calls if module_id == row["module_id"]} == set(DEFAULT_PROBE_PROFILES) for row in nonroot_modules)
    assert any(warning["code"] == "high_dependency_risk" for warnings in report["layer_warnings"].values() for warning in warnings)
    assert all("user_label" not in probe["measurement"] for probe in report["probes"])


def test_wrong_module_removal_or_missing_recovery_metrics_remain_unknown(tmp_path):
    model = FakeModel()
    inventory = capture_inventory(model)
    database = tmp_path / "unknown.sqlite"
    persist_inventory(database, inventory)

    class IncompleteAdapter:
        def sandbox_layer_probe(self, module, profile, context):
            del module, profile, context
            return {
                "isolated_copy": True,
                "removal_performed": True,
                "removed_module_id": "different-snapshot-module",
                "measurements": {"unqualified": 1},
            }

    report = run_layer_probe_battery(model, inventory, database, IncompleteAdapter())
    assert report["status"] == "partial"
    assert report["unknown_probe_count"] == report["attempted_probe_count"]
    assert all(not item["isolated"] or item["status"] == "unknown" for item in report["probes"])
    assert all("requested_module_removal_not_verified" in item["limitations"] for item in report["probes"])


def test_live_model_mutation_fails_the_run_and_restores_state(tmp_path):
    model = FakeModel()
    inventory = capture_inventory(model)
    database = tmp_path / "mutating.sqlite"
    persist_inventory(database, inventory)
    before = model.state_dict()["blocks.0.attention.weight"].raw

    class MutatingAdapter:
        def sandbox_layer_probe(self, module, profile, context):
            del profile, context
            model.blocks._children["0"]._children["attention"].weight.raw = b"evil"
            return {
                "isolated_copy": True,
                "removal_performed": True,
                "removed_module_id": module["module_id"],
                "measurements": _good_measurement("output_sensitivity"),
            }

    report = run_layer_probe_battery(model, inventory, database, MutatingAdapter())
    assert report["status"] == "failed"
    assert report["transient_live_mutation_detected"] is True
    assert report["model_modified"] is False
    assert model.state_dict()["blocks.0.attention.weight"].raw == before


def test_unseen_architecture_evaluator_rejects_leakage_and_reports_metrics():
    gold = [
        {"group_id": "arch-new-a", "target_id": "m1", "role": "attention_component"},
        {"group_id": "arch-new-b", "target_id": "m2", "role": "normalization"},
    ]
    predicted = [
        {"group_id": "arch-new-a", "target_id": "m1", "role": "attention_component"},
        {"group_id": "arch-new-b", "target_id": "m2", "role": "attention_component"},
    ]
    universe = [
        {"group_id": "arch-new-a", "target_id": "m1"},
        {"group_id": "arch-new-b", "target_id": "m2"},
    ]
    result = evaluate_unseen_architectures(gold, predicted, dataset_id="heldout-v1", universe_targets=universe, held_out_group_ids=["arch-new-a", "arch-new-b"], training_group_ids=["arch-known"])
    assert result["split_name"] == "unseen_architecture_holdout"
    assert result["metrics"]["per_role"]["attention_component"]["tp"] == 1
    assert result["metrics"]["per_role"]["attention_component"]["fp"] == 1
    assert result["metrics"]["target_coverage"] == 1.0
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_unseen_architectures(gold, predicted, dataset_id="bad", universe_targets=universe, held_out_group_ids=["arch-new-a"], training_group_ids=["arch-new-a"])


def test_database_initializer_is_idempotent(tmp_path):
    database = tmp_path / "empty.sqlite"
    initialize_database(database)
    initialize_database(database)
    connection = sqlite3.connect(database)
    assert connection.execute("SELECT COUNT(*) FROM schema_versions").fetchone()[0] == 1
    connection.close()


def test_orchestrator_captures_unknown_architecture_without_promoting_mutation(tmp_path):
    model = FakeModel()
    report = build_model_inventory(
        model,
        tmp_path / "full.sqlite",
        source_revision="unseen-demo@abc123",
        isolated_model=copy.deepcopy(model),
        invoke=lambda candidate: candidate(1),
        capture_fx=False,
    )
    assert report["inventory_status"] == "complete"
    assert report["graphs"][0]["status"] == "measured"
    assert report["role_assertions"]
    assert report["model_modified"] is False
    assert report["runtime_permission_granted"] is False

    with pytest.raises(PermissionError, match="separate disposable"):
        build_model_inventory(model, tmp_path / "same-model.sqlite", isolated_model=model, capture_fx=False)


def test_sandbox_discovery_can_opt_in_to_inventory_database(tmp_path, monkeypatch):
    monkeypatch.setattr(model_mapping_discovery, "_state_fingerprint", _state_fingerprint)
    monkeypatch.setattr(experiment_pipeline, "_state_fingerprint", model_mapping_discovery._state_fingerprint)
    adapter = SimpleNamespace(
        handle=SimpleNamespace(model=FakeModel(), tokenizer=None),
        sandboxed=True,
        sandbox_root=tmp_path,
    )
    inventory_path = tmp_path / "models.sqlite"
    context = {"sandbox": True, "sandbox_root": tmp_path, "model_inventory_db_path": inventory_path}
    report = sandbox_discovery.run(adapter, context)
    assert report["model_inventory"]["inventory_status"] == "complete"
    assert inventory_path.exists()
    assert report["runtime_mapping_promoted"] is False

    with pytest.raises(PermissionError, match="inside sandbox_root"):
        sandbox_discovery.run(adapter, {"sandbox": True, "sandbox_root": tmp_path, "model_inventory_db_path": tmp_path.parent / "outside.sqlite"})
