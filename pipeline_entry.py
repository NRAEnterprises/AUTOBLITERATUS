# abliterate/tuning/103_pipeline_entry.py
#
# Tuning Type 103 of 103:
# Pipeline entry.
#
# Runtime toggle: not applicable. This is the entry point.
#
# Effect:
#   Loads the selector, constructs every tuning object whose toggle is on,
#   hands the orchestrator the dependency graph, and runs the pipeline. This
#   is the only file the user invokes.

from __future__ import annotations

from abliterate.tuning.cli_selector import main as cli_main
from abliterate.tuning.runtime_selector import RuntimeSelector
from abliterate.tuning.toggle_registry import TOGGLES
from abliterate.tuning.tuning_orchestrator import TuningOrchestrator


DEPENDENCIES = {
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


def run(argv=None):
    args = cli_main(argv)
    selector = RuntimeSelector(toggles=args["toggles"])

    # Placeholder run_one: dispatches to the actual tuning callable registered
    # for the toggle id. In the assembled tool, this is a lookup from
    # toggle id to instantiated tuning object.
    def run_one(toggle_id: str, context: dict):
        if not selector.is_enabled(toggle_id):
            return
        # Instantiate and call the tuning object registered for this id.
        return

    orchestrator = TuningOrchestrator(
        registry=TOGGLES,
        dependencies=DEPENDENCIES,
        run_one=run_one,
    )
    return orchestrator.run(context=args)


if __name__ == "__main__":
    run()
