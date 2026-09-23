"""Smoke tests for the toggle registry."""

from __future__ import annotations

from autobliteratus.tuning.toggle_registry import TOGGLES, enabled_toggles


def test_registry_not_empty():
    assert TOGGLES


def test_every_toggle_has_id_and_default():
    for toggle_id, meta in TOGGLES.items():
        assert meta.get("id", toggle_id) == toggle_id
        assert "default" in meta
        assert isinstance(meta["default"], bool)


def test_enabled_toggles_respects_overrides():
    overrides = {tid: not meta["default"] for tid, meta in TOGGLES.items()}
    resolved = enabled_toggles(overrides)
    for tid, value in overrides.items():
        assert resolved[tid] is value
