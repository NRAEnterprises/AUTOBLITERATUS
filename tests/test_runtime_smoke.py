from autobliteratus.tuning.runtime_selector import RuntimeSelector
from autobliteratus.tuning.tuning_orchestrator import TuningOrchestrator
from autobliteratus.tuning.toggle_registry import TOGGLES


def test_imports() -> None:
    import autobliteratus  # noqa: F401
    import autobliteratus.tuning  # noqa: F401


def test_runtime_selector() -> None:
    selector = RuntimeSelector(toggles={"demo.toggle": True})
    assert selector.is_enabled("demo.toggle") is True
    assert selector.is_enabled("demo.off") is False


def test_toggle_registry_not_empty() -> None:
    assert isinstance(TOGGLES, dict)
    assert len(TOGGLES) > 0


def test_orchestrator_runs_in_dependency_order() -> None:
    executed: list[str] = []

    def run_one(toggle_id: str, context: dict) -> None:
        executed.append(toggle_id)

    registry = {"a": {}, "b": {}, "c": {}}
    dependencies = {"b": ["a"], "c": ["b"]}
    context = {"_resolved_toggles": {"a": True, "b": True, "c": True}}

    report = TuningOrchestrator(
        registry=registry,
        dependencies=dependencies,
        run_one=run_one,
    ).run(context=context)

    assert executed == ["a", "b", "c"]
    assert report.executed == ["a", "b", "c"]
