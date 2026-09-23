"""Smoke tests for the orchestrator."""

from __future__ import annotations

from autobliteratus.tuning.tuning_orchestrator import TuningOrchestrator


def test_skips_disabled_toggles():
    registry = {"a": {"default": True}, "b": {"default": True}}
    deps = {}
    calls = []

    def run_one(tid, ctx):
        calls.append(tid)

    orch = TuningOrchestrator(registry, deps, run_one)
    report = orch.run(context={"_resolved_toggles": {"a": True, "b": False}})

    assert calls == ["a"]
    assert report.executed == ["a"]
    assert report.skipped == ["b"]


def test_failed_toggle_is_recorded():
    registry = {"a": {"default": True}}
    deps = {}

    def run_one(tid, ctx):
        raise ValueError("boom")

    orch = TuningOrchestrator(registry, deps, run_one)
    report = orch.run(context={"_resolved_toggles": {"a": True}})

    assert report.executed == []
    assert report.failed[0]["id"] == "a"
