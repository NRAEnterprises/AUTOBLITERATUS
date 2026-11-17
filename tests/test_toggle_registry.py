"""Smoke tests for the toggle registry."""

from __future__ import annotations

import pytest

from autobliteratus.tuning.toggle_registry import TOGGLES, enabled_toggles


def test_registry_not_empty():
    assert TOGGLES


def test_every_toggle_has_id_and_default():
    for toggle_id, meta in TOGGLES.items():
        assert meta.get("id", toggle_id) == toggle_id
        assert "default" in meta
        assert isinstance(meta["default"], bool)


def test_enabled_toggles_respects_overrides():
    overrides = {
        tid: not meta["default"] if meta["available"] else False
        for tid, meta in TOGGLES.items()
    }
    resolved = enabled_toggles(overrides)
    for tid, value in overrides.items():
        assert resolved[tid] is value


@pytest.mark.parametrize(
    "prototype",
    [tid for tid, meta in TOGGLES.items() if not meta["available"]],
)
def test_every_unavailable_prototype_cannot_be_enabled(prototype):
    with pytest.raises(ValueError, match="prototype"):
        enabled_toggles({prototype: True})


def test_core_pipeline_stages_have_real_callable_contracts():
    assert TOGGLES["tuning.architecture_detection"]["available"] is True
    assert TOGGLES["tuning.refusal_direction_ablation"]["available"] is True
    assert TOGGLES["tuning.calibrated_quantization"]["available"] is True
    assert TOGGLES["tuning.model_export"]["available"] is True
    assert TOGGLES["tuning.memory_footprint_auto"]["available"] is True
    assert TOGGLES["tuning.reproducibility_auto"]["available"] is True


def test_selected_high_priority_stages_have_callable_contracts_and_default_off():
    for toggle_id in (
        "tuning.anti_router_collapse",
        "tuning.multi_objective_auto",
        "tuning.auto_security_scanning",
    ):
        assert TOGGLES[toggle_id]["available"] is True
        assert TOGGLES[toggle_id]["default"] is False


def test_baseline_regression_and_audit_stages_are_callable_and_default_off():
    for toggle_id in (
        "tuning.baseline_auto_pre_post",
        "tuning.benchmark_regression_auto",
        "tuning.audit_trail_auto",
    ):
        assert TOGGLES[toggle_id]["available"] is True
        assert TOGGLES[toggle_id]["default"] is False
