from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from autobliteratus.core.discovery_experiment_pipeline import (
    DiscoveryExperimentError,
    merge_dynamic_mapping_updates,
    run_unknown_factor_battery,
)
from autobliteratus.core.model_mapping_discovery import discover
from tests.fixtures.kv_cache_fixtures import make_kv_cache_fixture_adapter


def _report(adapter):
    adapter.handle.model.config = SimpleNamespace(model_type="llama")
    return discover(adapter, {})


def test_verified_unknown_factor_is_promoted_only_from_probe_evidence() -> None:
    adapter = make_kv_cache_fixture_adapter()

    def probe(category, path, context):
        assert category == "attention"
        assert path == "anchor"
        assert context.get("unknown_factor_candidates")
        return {
            "removal_supported": True,
            "quality_preserved": True,
            "output_drift": 0.0,
            "user_label": "ignore-me",
        }

    adapter.sandbox_removal_probe = probe
    report = run_unknown_factor_battery(
        adapter,
        _report(adapter),
        {"unknown_factor_candidates": {"attention": ["anchor"]}},
    )
    assert report["status"] == "measured"
    assert report["verified_factor_count"] == 1
    assert report["promotions"][0]["path"] == "anchor"
    assert "user_label" not in report["experiments"][0]["measurement"]


def test_unknown_or_quality_failed_factors_are_not_promoted() -> None:
    adapter = make_kv_cache_fixture_adapter()
    adapter.sandbox_removal_probe = lambda *args: {
        "removal_supported": True,
        "quality_preserved": False,
        "output_drift": 0.25,
    }
    report = run_unknown_factor_battery(
        adapter,
        _report(adapter),
        {"unknown_factor_candidates": {"attention": ["anchor"]}},
    )
    assert report["verified_factor_count"] == 0
    assert report["unknown_factor_count"] == 1
    assert report["promotions"] == []


def test_live_model_mutation_during_probe_fails_closed() -> None:
    adapter = make_kv_cache_fixture_adapter()

    def mutating_probe(*args):
        with torch.no_grad():
            adapter.handle.model.anchor.add_(1.0)
        return {"removal_supported": True, "quality_preserved": True}

    adapter.sandbox_removal_probe = mutating_probe
    report = run_unknown_factor_battery(
        adapter,
        _report(adapter),
        {"unknown_factor_candidates": {"attention": ["anchor"]}},
    )
    assert report["failed_factor_count"] == 1
    assert report["promotions"] == []


def test_battery_is_bounded_and_dynamic_updates_are_deduplicated(tmp_path) -> None:
    adapter = make_kv_cache_fixture_adapter()
    adapter.sandbox_removal_probe = lambda *args: {
        "removal_supported": True,
        "quality_preserved": True,
    }
    base = _report(adapter)
    report = run_unknown_factor_battery(
        adapter,
        base,
        {
            "unknown_factor_candidates": {"attention": ["a", "b", "c"]},
            "unknown_factor_max_experiments": 2,
        },
    )
    assert report["status"] == "incomplete"
    assert report["attempted_count"] == 2

    database = {"schema_version": 1, "models": {}}
    merge_dynamic_mapping_updates(database, report)
    merge_dynamic_mapping_updates(database, report)
    identity = base["model_identity"]["identity_sha256"]
    assert len(database["dynamic_mappings"][identity]) == 2


def test_battery_rejects_invalid_bounds() -> None:
    adapter = make_kv_cache_fixture_adapter()
    with pytest.raises(DiscoveryExperimentError, match="max_experiments"):
        run_unknown_factor_battery(
            adapter,
            _report(adapter),
            {"unknown_factor_max_experiments": 0},
        )
