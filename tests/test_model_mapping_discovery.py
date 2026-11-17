from __future__ import annotations

import json
from types import SimpleNamespace

from autobliteratus.core.model_mapping_discovery import discover, run, update_database
from tests.fixtures.kv_cache_fixtures import make_kv_cache_fixture_adapter


def _adapter():
    adapter = make_kv_cache_fixture_adapter()
    adapter.handle.model.config = SimpleNamespace(model_type="llama")
    return adapter


def test_discovery_runs_battery_and_keeps_unverified_areas_unknown() -> None:
    report = discover(_adapter(), {})

    assert report["stage_id"] == "tuning.model_mapping_discovery"
    assert report["status"] == "measured"
    assert report["model_modified"] is False
    assert report["mutation_detected"] is False
    assert report["unknown_area_count"] >= 1
    assert report["areas"]["layer_stack"]["status"] == "unknown"
    assert report["areas"]["kv_cache"]["status"] == "verified"
    assert {item["code"] for item in report["experiments"]} >= {
        "config_identity",
        "state_inventory",
        "candidate_path_resolution",
        "weight_shape_validation",
        "cache_api_contract",
        "cache_output_stability",
        "state_mutation_guard",
    }


def test_discovery_persists_atomic_evidence_history(tmp_path) -> None:
    database_path = tmp_path / "mappings.json"
    context = {"model_mapping_db_path": database_path}

    report = run(_adapter(), context)
    run(_adapter(), context)

    database = json.loads(database_path.read_text(encoding="utf-8"))
    identity = report["model_identity"]["identity_sha256"]
    assert database["schema_version"] == 1
    assert len(database["models"][identity]) == 2
    assert report["database_path"] == str(database_path)
    assert not list(tmp_path.glob(".*.tmp"))


def test_database_rejects_invalid_existing_content(tmp_path) -> None:
    database_path = tmp_path / "mappings.json"
    database_path.write_text("not json", encoding="utf-8")
    report = discover(_adapter(), {})

    try:
        update_database(database_path, report)
    except ValueError as exc:
        assert "valid JSON" in str(exc)
    else:
        raise AssertionError("invalid mapping database must be rejected")
