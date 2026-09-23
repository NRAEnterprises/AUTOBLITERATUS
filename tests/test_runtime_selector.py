"""Smoke tests for the runtime selector."""

from __future__ import annotations

import pytest

from autobliteratus.tuning.runtime_selector import RuntimeSelector


def test_is_enabled_true():
    s = RuntimeSelector(toggles={"a": True})
    assert s.is_enabled("a")


def test_is_enabled_false_for_missing():
    s = RuntimeSelector(toggles={})
    assert not s.is_enabled("a")


def test_require_raises_when_disabled():
    s = RuntimeSelector(toggles={"a": False})
    with pytest.raises(RuntimeError):
        s.require("a")


def test_require_passes_when_enabled():
    s = RuntimeSelector(toggles={"a": True})
    s.require("a")
