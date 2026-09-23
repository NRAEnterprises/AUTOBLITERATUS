"""Top-level pipeline entry. Loads the model, builds the adapter, hands the
real registry to the orchestrator, runs the enabled toggles in dependency
order, returns a report."""

from __future__ import annotations

from typing import Any


DEPENDENCIES: dict[str, list[str]] = {
    "tuning.post_ablation_capability":           ["tuning.auto_ablated_capability",
                                                  "tuning.baseline_auto_pre_post",
                                                  "tuning.marker_creation"],
    "tuning.architecture_aware_capability":      ["tuning.architecture_detection",
                                                  "tuning.auto_ablated_capability",
                                                  "tuning.baseline_auto_pre_post"],
    "tuning.quantization_calibrated_capability": ["tuning.quantization_scheme_selection",
                                                  "tuning.auto_ablated_capability",
                                                  "tuning.baseline_auto_pre_post"],
    "tuning.quantization_aware_capability":      ["tuning.quantization_scheme_selection"],
    "tuning.method_auto_selection":              ["tuning.architecture_detection"],
    "tuning.yaml_config_auto":                   ["tuning.architecture_detection",
                                                  "tuning.auto_ablated_capability"],
    "tuning.auto_rollback":                      ["tuning.marker_creation"],
    "tuning.export_format_auto":                 ["tuning.model_card_generation_auto",
                                                  "tuning.provenance_recording_auto",
                                                  "tuning.audit_trail_auto"],
}


def run_pipeline(
    model_path: str,
    work_dir: str = "./abliterate_workspace",
    quantization: str = "none",
    target_frameworks: list[str] | None = None,
    deployment_target: str = "server",
    enable: list[str] | None = None,
    disable: list[str] | None = None,
    dry_run: bool = False,
    **kwargs: Any,
) -> dict:
    from autobliteratus.core.adapters import TuningAdapter
    from autobliteratus.tuning.dispatch import build_registry
    from autobliteratus.tuning.runtime_selector import RuntimeSelector
    from autobliteratus.tuning.toggle_registry import TOGGLES, enabled_toggles
    from autobliteratus.tuning.tuning_orchestrator import TuningOrchestrator

    overrides: dict[str, bool] = {tid: True for tid in (enable or [])}
    overrides.update({tid: False for tid in (disable or [])})
    resolved = enabled_toggles(overrides)
    selector = RuntimeSelector(toggles=resolved)

    adapter = TuningAdapter.from_path(model_path)

    context: dict[str, Any] = {
        "model_path": model_path,
        "work_dir": work_dir,
        "quantization": quantization,
        "target_frameworks": target_frameworks or ["transformers"],
        "deployment_target": deployment_target,
        "dry_run": dry_run,
        "_resolved_toggles": resolved,
    }

    registry = build_registry(adapter, context)

    def run_one(toggle_id: str, ctx: dict):
        if not selector.is_enabled(toggle_id):
            return None
        fn = registry.get(toggle_id)
        if fn is None:
            return None
        return fn(adapter, ctx)

    orchestrator = TuningOrchestrator(
        registry=TOGGLES,
        dependencies=DEPENDENCIES,
        run_one=run_one,
    )
    report = orchestrator.run(context=context)

    return {
        "success": not report.failed,
        "executed": report.executed,
        "skipped": report.skipped,
        "failed": report.failed,
        "model_summary": adapter.handle.summary(),
    }
