from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from autobliteratus.core.model_mapping_discovery import TOGGLE as DISCOVERY_CORE_TOGGLE
from autobliteratus.discovery_pipeline import PIPELINE_ID, run
from autobliteratus.tuning.toggle_registry import TOGGLES
from tests.fixtures.kv_cache_fixtures import make_kv_cache_fixture_adapter


def test_discovery_pipeline_requires_explicit_sandbox(tmp_path) -> None:
    with pytest.raises(PermissionError, match="sandbox=True"):
        run(make_kv_cache_fixture_adapter(), {"sandbox_root": tmp_path})


def test_discovery_pipeline_rejects_runtime_and_mutating_contexts(tmp_path) -> None:
    adapter = make_kv_cache_fixture_adapter()
    with pytest.raises(PermissionError, match="live or runtime"):
        run(adapter, {"sandbox": True, "sandbox_root": tmp_path, "runtime_mode": "production"})
    with pytest.raises(PermissionError, match="model mutation"):
        run(adapter, {"sandbox": True, "sandbox_root": tmp_path, "allow_model_mutation": True})


def test_discovery_pipeline_keeps_database_inside_sandbox(tmp_path) -> None:
    adapter = make_kv_cache_fixture_adapter()
    adapter.sandboxed = True
    adapter.sandbox_root = tmp_path
    report = run(adapter, {"sandbox": True, "sandbox_root": tmp_path})

    database_path = tmp_path / "model-mappings.json"
    assert report["pipeline_id"] == PIPELINE_ID
    assert report["runtime_pipeline_touched"] is False
    assert report["runtime_mapping_promoted"] is False
    assert report["model_modified"] is False
    assert database_path.exists()
    assert json.loads(database_path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_discovery_pipeline_persists_verified_dynamic_mapping(tmp_path) -> None:
    adapter = make_kv_cache_fixture_adapter()
    adapter.handle.model.config = SimpleNamespace(model_type="llama")
    adapter.sandboxed = True
    adapter.sandbox_root = tmp_path
    adapter.sandbox_removal_probe = lambda *args: {
        "removal_supported": True,
        "quality_preserved": True,
    }

    report = run(
        adapter,
        {
            "sandbox": True,
            "sandbox_root": tmp_path,
            "unknown_factor_candidates": {"attention": ["anchor"]},
        },
    )

    database = json.loads((tmp_path / "model-mappings.json").read_text(encoding="utf-8"))
    identity = report["model_identity"]["identity_sha256"]
    assert report["unknown_factor_battery"]["verified_factor_count"] == 1
    assert database["dynamic_mappings"][identity][0]["path"] == "anchor"


def test_discovery_pipeline_rejects_an_unbound_adapter(tmp_path) -> None:
    with pytest.raises(PermissionError, match="adapter constructed"):
        run(make_kv_cache_fixture_adapter(), {"sandbox": True, "sandbox_root": tmp_path})


def test_runtime_registry_does_not_register_discovery_stage() -> None:
    assert DISCOVERY_CORE_TOGGLE["id"] == "tuning.model_mapping_discovery"
    assert "tuning.model_mapping_discovery" not in TOGGLES
