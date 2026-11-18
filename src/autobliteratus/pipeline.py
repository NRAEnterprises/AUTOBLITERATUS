"""Top-level pipeline: load, transform, validate, and export a separate model."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

DEPENDENCIES: dict[str, list[str]] = {
    "tuning.refusal_direction_ablation": ["tuning.architecture_detection"],
    "tuning.model_export": ["tuning.refusal_direction_ablation"],
    "tuning.calibrated_quantization": ["tuning.refusal_direction_ablation"],
    "tuning.weight_fingerprint_auto": [
        "tuning.refusal_direction_ablation",
        "tuning.calibrated_quantization",
    ],
    "tuning.provenance_recording_auto": [
        "tuning.refusal_direction_ablation",
        "tuning.calibrated_quantization",
        "tuning.weight_fingerprint_auto",
    ],
    "tuning.model_card_generation_auto": [
        "tuning.architecture_detection",
        "tuning.refusal_direction_ablation",
        "tuning.calibrated_quantization",
        "tuning.provenance_recording_auto",
    ],
    "tuning.memory_footprint_auto": [
        "tuning.refusal_direction_ablation",
        "tuning.calibrated_quantization",
    ],
    "tuning.reproducibility_auto": [
        "tuning.refusal_direction_ablation",
        "tuning.calibrated_quantization",
    ],
    "tuning.benchmark_regression_auto": ["tuning.baseline_auto_pre_post"],
    "tuning.version_compatibility_auto": ["tuning.model_export"],
    "tuning.auto_ablated_capability": ["tuning.refusal_direction_ablation"],
    "tuning.latency_auto": ["tuning.refusal_direction_ablation", "tuning.calibrated_quantization"],
    "tuning.batch_size_throughput_auto": ["tuning.latency_auto", "tuning.memory_footprint_auto"],
    "tuning.resource_budget_gate": [
        "tuning.memory_footprint_auto",
        "tuning.latency_auto",
        "tuning.batch_size_throughput_auto",
    ],
}


def run_pipeline(
    model_path: str,
    model_revision: str | None = None,
    work_dir: str = "./abliterate_workspace",
    output_dir: str | None = None,
    quantization: str = "none",
    target_frameworks: list[str] | None = None,
    deployment_target: str = "server",
    enable: list[str] | None = None,
    disable: list[str] | None = None,
    dry_run: bool = False,
    allow_mutation: bool = False,
    ablation_strength: float = 1.0,
    ablation_component: str = "attn.out",
    trust_remote_code: bool = False,
    calibration_data: str | None = None,
    calibration_samples: int = 128,
    calibration_max_length: int = 256,
    calibration_momentum: float = 0.9,
    calibration_streamline: bool = True,
    device: str = "auto",
    dtype: str = "auto",
    device_map: str | None = None,
    max_memory: dict[str | int, str | int] | None = None,
    offload_folder: str | None = None,
    max_shard_size: str = "5GB",
    plan_only: bool = False,
    stage_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run supported tuning stages and write the result to a new directory.

    Model loading is read-only with respect to ``model_path``. Transformations
    are applied to the in-memory model; the export stage publishes only to the
    disjoint ``output_dir`` after validating the temporary export.
    """
    from autobliteratus.core.adapters import TuningAdapter
    from autobliteratus.tuning._96_model_export import _validate_disjoint_paths
    from autobliteratus.tuning.dispatch import build_registry
    from autobliteratus.tuning.mutation_firewall import MUTATING_STAGE_IDS
    from autobliteratus.tuning.mutation_transaction import MutationTransaction
    from autobliteratus.tuning.runtime_selector import RuntimeSelector
    from autobliteratus.tuning.toggle_registry import TOGGLES, enabled_toggles
    from autobliteratus.tuning.tuning_orchestrator import TuningOrchestrator

    supported_quantization = {
        "none",
        "quanto-int4",
        "quanto-int8",
        "quanto-int4-weight-only",
        "quanto-int8-weight-only",
    }
    if quantization not in supported_quantization:
        raise ValueError(
            f"Unsupported quantization mode {quantization!r}; choose one of {sorted(supported_quantization)}."
        )
    if model_revision is not None and (not isinstance(model_revision, str) or not model_revision.strip()):
        raise ValueError("model_revision must be a non-empty immutable Hugging Face revision string.")
    quantization_id = "tuning.calibrated_quantization"
    requested_enable = set(enable or [])
    requested_disable = set(disable or [])
    unknown = sorted((requested_enable | requested_disable) - set(TOGGLES))
    if unknown:
        raise ValueError(f"Unknown tuning stage IDs: {unknown}")
    unavailable = sorted(
        toggle_id
        for toggle_id in requested_enable | requested_disable
        if not TOGGLES[toggle_id].get("available", False)
    )
    if unavailable and not plan_only:
        raise ValueError(
            "Requested tuning stages are prototypes without a callable implementation: "
            f"{unavailable}. They cannot be enabled or disabled as runtime stages."
        )
    overlap = sorted(requested_enable & requested_disable)
    if overlap:
        raise ValueError(f"A tuning stage cannot be both enabled and disabled: {overlap}")
    requested_frameworks = list(target_frameworks or ["transformers"])
    unsupported_frameworks = sorted(set(requested_frameworks) - {"transformers"})
    if unsupported_frameworks:
        raise ValueError(
            "Only the Transformers export/runtime is implemented; unsupported target framework(s): "
            f"{unsupported_frameworks}."
        )
    if not requested_frameworks:
        raise ValueError("At least one target framework is required.")
    allowed_stage_options = {
        "routing_probe_texts",
        "routing_max_samples",
        "routing_max_length",
        "routing_entropy_threshold",
        "multi_objective_candidates",
        "multi_objective_objectives",
        "multi_objective_weights",
        "security_scan_max_abs",
        "evaluation_texts",
        "evaluation_data",
        "evaluation_max_samples",
        "evaluation_max_length",
        "regression_metric_directions",
        "regression_tolerance_ratio",
        "regression_tolerances",
        "compatibility_loaders",
        "tokenizer_before",
        "tokenizer_after",
        "tokenization_corpus",
        "tokenization_max_samples",
        "capability_axes",
        "latency_prompts",
        "latency_max_new_tokens",
        "latency_warmups",
        "throughput_prompts",
        "throughput_batch_sizes",
        "throughput_max_new_tokens",
        "throughput_warmups",
        "throughput_timeout_seconds",
        "throughput_cancelled",
        "resource_budget",
        "kv_cache_prompts",
        "kv_cache_sequence_lengths",
        "kv_cache_batch_sizes",
        "kv_cache_modes",
        "kv_cache_max_new_tokens",
        "kv_cache_timeout_seconds",
        "kv_cache_cancelled",
        "kv_cache_output_tolerance",
        "topology_proposal_path",
        "topology_sandbox_root",
        "topology_expected_identity",
        "mapping_capability",
        "mapping_expected_identity",
    }
    if stage_options is not None and not isinstance(stage_options, dict):
        raise TypeError("stage_options must be a mapping of supported stage option names to values.")
    unknown_stage_options = sorted(set(stage_options or {}) - allowed_stage_options)
    if unknown_stage_options:
        raise ValueError(f"Unknown stage_options keys: {unknown_stage_options}")
    if deployment_target not in {"server", "desktop", "edge", "mobile"}:
        raise ValueError(f"Unsupported deployment target {deployment_target!r}.")
    if not math.isfinite(ablation_strength) or not 0.0 < ablation_strength <= 1.0:
        raise ValueError("ablation_strength must be finite and in the interval (0, 1].")
    if ablation_component not in {"attn.out", "mlp.down"}:
        raise ValueError("ablation_component must be 'attn.out' or 'mlp.down'.")
    if not dry_run and "tuning.model_export" in requested_disable:
        raise ValueError("Model export cannot be disabled for a modifying run; use --dry-run for inspection only.")
    if quantization == "none" and quantization_id in requested_enable:
        raise ValueError("Select --quantization quanto-int4 or quanto-int8 to enable calibrated quantization.")
    if quantization != "none" and quantization_id in requested_disable:
        raise ValueError("A quantization mode cannot be selected while its stage is explicitly disabled.")
    for name, values in (("enable", enable or []), ("disable", disable or [])):
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate toggle IDs are not allowed in --{name}.")
    if device_map not in {None, "auto", "balanced", "balanced_low_0", "sequential"}:
        raise ValueError(f"Unsupported device_map strategy {device_map!r}.")
    if max_memory is not None and not isinstance(max_memory, dict):
        raise ValueError("max_memory must be a device-to-memory mapping.")
    if (max_memory is not None or offload_folder is not None) and device_map is None:
        device_map = "auto"
    if not re.fullmatch(r"[1-9][0-9]*(?:\.[0-9]+)?(?:B|KB|MB|GB|TB|KiB|MiB|GiB|TiB)", max_shard_size):
        raise ValueError("max_shard_size must be a positive size such as '512MiB' or '5GB'.")

    effective_output_dir = str(
        Path(output_dir).expanduser() if output_dir else Path(work_dir).expanduser() / "modified_model"
    )
    activation_quantized = quantization in {"quanto-int4", "quanto-int8"}
    if quantization != "none":
        if activation_quantized and not calibration_data:
            raise ValueError("Calibrated quantization requires --calibration-data with representative text.")
        if activation_quantized and calibration_samples < 1:
            raise ValueError("calibration_samples must be at least 1.")
        if activation_quantized and calibration_max_length < 2:
            raise ValueError("calibration_max_length must be at least 2.")
        if activation_quantized and not 0.0 <= calibration_momentum <= 1.0:
            raise ValueError("calibration_momentum must be in [0, 1].")
        if activation_quantized:
            calibration_path = Path(calibration_data).expanduser()
            if not calibration_path.is_file():
                raise FileNotFoundError(f"Calibration data file does not exist: {calibration_path}")
    elif calibration_data:
        calibration_path = Path(calibration_data).expanduser()
        if not calibration_path.is_file():
            raise FileNotFoundError(f"Calibration data file does not exist: {calibration_path}")
    if not dry_run and "tuning.model_export" not in requested_disable:
        _validate_disjoint_paths(model_path, Path(effective_output_dir))
        if Path(effective_output_dir).exists() or Path(effective_output_dir).is_symlink():
            raise FileExistsError(
                f"Output directory already exists: {effective_output_dir}. Choose a new path."
            )
    if offload_folder:
        _validate_disjoint_paths(model_path, Path(offload_folder).expanduser())
        if Path(effective_output_dir).expanduser().resolve() == Path(offload_folder).expanduser().resolve():
            raise ValueError("offload_folder must be separate from output_dir.")

    overrides: dict[str, bool] = {tid: True for tid in (enable or [])}
    overrides.update({tid: False for tid in (disable or [])})
    baseline_id = "tuning.baseline_auto_pre_post"
    regression_id = "tuning.benchmark_regression_auto"
    if regression_id in requested_enable:
        if baseline_id in requested_disable:
            raise ValueError("tuning.benchmark_regression_auto requires tuning.baseline_auto_pre_post.")
        overrides[baseline_id] = True
    if quantization != "none":
        overrides[quantization_id] = True
    if dry_run:
        # A dry run may load/inspect the model, but never mutates or publishes it.
        overrides["tuning.refusal_direction_ablation"] = False
        overrides["tuning.model_export"] = False
        overrides[quantization_id] = False

    resolved = enabled_toggles(overrides, allow_unavailable=plan_only)
    mutating_stage_ids = sorted(
        stage_id for stage_id in MUTATING_STAGE_IDS if resolved.get(stage_id, False)
    )
    if mutating_stage_ids and not dry_run and not plan_only:
        if allow_mutation is not True:
            raise PermissionError(
                "Mutation-capable stages are disabled by default; pass allow_mutation=True "
                "or --allow-mutation for an explicitly authorized transactional run."
            )
        if not resolved.get("tuning.model_export", False):
            raise ValueError(
                "A modifying run must explicitly enable tuning.model_export so the "
                "validated artifact is published only after the transaction succeeds."
            )
    dependencies = {key: list(value) for key, value in DEPENDENCIES.items()}
    evaluation_enabled = resolved.get(baseline_id, False) or resolved.get(regression_id, False)
    if evaluation_enabled:
        evaluation_texts = (stage_options or {}).get("evaluation_texts")
        evaluation_data = (stage_options or {}).get("evaluation_data")
        if bool(evaluation_texts) == bool(evaluation_data):
            raise ValueError("Provide exactly one of stage_options.evaluation_texts or stage_options.evaluation_data.")
        if evaluation_texts is not None and (
            not isinstance(evaluation_texts, (list, tuple))
            or not all(isinstance(text, str) for text in evaluation_texts)
            or not any(text.strip() for text in evaluation_texts)
        ):
            raise TypeError("stage_options.evaluation_texts must be a non-empty list of strings.")
        if evaluation_data is not None and not Path(evaluation_data).expanduser().is_file():
            raise FileNotFoundError(f"Evaluation data file does not exist: {evaluation_data}")
        for option, minimum in (("evaluation_max_samples", 1), ("evaluation_max_length", 2)):
            value = (stage_options or {}).get(option, 64 if option == "evaluation_max_samples" else 256)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"stage_options.{option} must be an integer of at least {minimum}.")
        directions = (stage_options or {}).get(
            "regression_metric_directions",
            {"mean_nll": "min", "perplexity": "min", "forward_tokens_per_second": "max"},
        )
        if not isinstance(directions, dict) or not directions or any(
            not isinstance(name, str) or direction not in {"min", "max"}
            for name, direction in directions.items()
        ):
            raise ValueError("stage_options.regression_metric_directions must map metric names to 'min' or 'max'.")
        tolerance = (stage_options or {}).get("regression_tolerance_ratio", 0.02)
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(float(tolerance)) or not 0.0 <= tolerance <= 1.0:
            raise ValueError("stage_options.regression_tolerance_ratio must be finite and in [0, 1].")
        per_metric_tolerances = (stage_options or {}).get("regression_tolerances", {})
        if not isinstance(per_metric_tolerances, dict):
            raise TypeError("stage_options.regression_tolerances must map metric names to relative tolerances.")
        if any(
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= value <= 1.0
            for name, value in per_metric_tolerances.items()
        ):
            raise ValueError("Each regression_tolerances value must be finite and in [0, 1].")

    if resolved.get(baseline_id, False):
        for modifier in ("tuning.refusal_direction_ablation", quantization_id):
            if resolved.get(modifier, False):
                dependencies[modifier] = list(dict.fromkeys(dependencies.get(modifier, []) + [baseline_id]))
    if resolved.get(regression_id, False):
        dependencies[regression_id] = [baseline_id] + [
            modifier
            for modifier in ("tuning.refusal_direction_ablation", quantization_id)
            if resolved.get(modifier, False)
        ]

    audit_id = "tuning.audit_trail_auto"
    audit_dir = Path(work_dir).expanduser() / "audit_trails"
    if resolved.get(audit_id, False):
        _validate_disjoint_paths(model_path, audit_dir)
        audit_target = audit_dir.resolve()
        export_target = Path(effective_output_dir).expanduser().resolve()
        if (
            audit_target == export_target
            or audit_target.is_relative_to(export_target)
            or export_target.is_relative_to(audit_target)
        ):
            raise ValueError("The audit directory must be separate from the exported model directory.")

    export_id = "tuning.model_export"
    if export_id in resolved:
        dependencies[export_id] = [
            toggle_id
            for toggle_id, is_enabled in resolved.items()
            if is_enabled and toggle_id != export_id
        ]
    missing_dependencies = sorted(
        {dep for stage, deps in dependencies.items() for dep in deps if dep not in TOGGLES}
    )
    if missing_dependencies:
        raise ValueError(f"Dependency graph references unknown toggle IDs: {missing_dependencies}")
    operation_order = TuningOrchestrator(
        registry=TOGGLES,
        dependencies=dependencies,
        run_one=lambda _toggle_id, _context: None,
    )._topological_order()

    if plan_only:
        unavailable_enabled = [
            toggle_id
            for toggle_id, is_enabled in resolved.items()
            if is_enabled and not TOGGLES[toggle_id].get("available", False)
        ]
        planned = [
            toggle_id
            for toggle_id in operation_order
            if resolved.get(toggle_id, False) and TOGGLES[toggle_id].get("available", False)
        ]
        skipped = [toggle_id for toggle_id, is_enabled in resolved.items() if not is_enabled]
        return {
            "success": not unavailable_enabled,
            "plan_only": True,
            "executed": [],
            "planned": planned,
            "skipped": skipped,
            "unavailable": unavailable_enabled,
            "failed": [],
            "model_summary": "Not loaded (--plan-only).",
            "model_modified": False,
            "output_dir": effective_output_dir,
            "plan": {
                "model_path": model_path,
                "model_revision": model_revision,
                "output_dir": effective_output_dir,
                "operation_graph": planned,
                "device": device,
                "device_map": device_map,
                "max_memory": max_memory,
                "offload_folder": offload_folder,
                "max_shard_size": max_shard_size,
                "unavailable_stages": unavailable_enabled,
                "evaluation_stages": [
                    stage for stage in (baseline_id, regression_id) if resolved.get(stage, False)
                ],
                "audit_dir": str(audit_dir) if resolved.get(audit_id, False) else None,
            },
        }

    selector = RuntimeSelector(toggles=resolved)
    adapter = TuningAdapter.from_path(
        model_path,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        max_memory=max_memory,
        offload_folder=offload_folder,
        revision=model_revision,
    )

    context: dict[str, Any] = {
        "model_path": model_path,
        "model_revision": model_revision,
        "work_dir": work_dir,
        "output_dir": effective_output_dir,
        "quantization": quantization,
        "target_frameworks": requested_frameworks,
        "deployment_target": deployment_target,
        "dry_run": dry_run,
        "allow_mutation": allow_mutation,
        "ablation_strength": ablation_strength,
        "ablation_component": ablation_component,
        "trust_remote_code": trust_remote_code,
        "device": device,
        "dtype": dtype,
        "device_map": device_map,
        "max_memory": max_memory,
        "offload_folder": offload_folder,
        "max_shard_size": max_shard_size,
        "calibration_data": str(Path(calibration_data).expanduser()) if calibration_data else None,
        "calibration_samples": calibration_samples,
        "calibration_max_length": calibration_max_length,
        "calibration_momentum": calibration_momentum,
        "calibration_streamline": calibration_streamline,
        "reproducibility_report": None,
        "memory_report": None,
        "model_modified": False,
        "_resolved_toggles": resolved,
    }
    context.update(stage_options or {})

    mutation_transaction = None
    if mutating_stage_ids and not dry_run:
        topology_source = context.get("topology_proposal_path")
        if topology_source is not None:
            from autobliteratus.core.topology_proposal import validate_topology_proposal

            expected_identity = context.get("topology_expected_identity")
            if not isinstance(expected_identity, dict) or not expected_identity:
                raise ValueError(
                    "topology_expected_identity must be provided when topology_proposal_path is used."
                )
            context["topology_proposal_report"] = validate_topology_proposal(
                topology_source,
                expected_identity=expected_identity,
                sandbox_root=context.get("topology_sandbox_root"),
            )
        mutation_transaction = MutationTransaction.begin(
            adapter,
            work_dir,
            model_revision=model_revision,
            stage="runtime-mutation",
        )
        context["mutation_transaction"] = mutation_transaction
        context["mutation_gate"] = {
            "status": "armed",
            "stages": mutating_stage_ids,
            "topology_proposal_validated": "topology_proposal_report" in context,
        }

    registry = build_registry(adapter, context)

    def run_one(toggle_id: str, ctx: dict[str, Any]) -> Any:
        if not selector.is_enabled(toggle_id):
            return None
        fn = registry.get(toggle_id)
        if fn is None:
            raise RuntimeError(
                f"Stage '{toggle_id}' is enabled but has no callable module-level run(adapter, context) implementation."
            )
        return fn(adapter, ctx)

    report = TuningOrchestrator(
        registry=TOGGLES,
        dependencies=dependencies,
        run_one=run_one,
    ).run(context=context)

    if context.get("_audit_requested"):
        from autobliteratus.tuning._89_audit_trail_auto_tuning import (
            finalize_pipeline_audit,
        )

        try:
            finalize_pipeline_audit(adapter, context, report)
        except Exception as exc:  # noqa: BLE001 - audit failure must become a failed stage in the final report.
            report.executed = [stage for stage in report.executed if stage != audit_id]
            report.failed.append(
                {"id": audit_id, "error_type": type(exc).__name__, "error": str(exc)}
            )

    mutation_transaction_report = None
    if mutation_transaction is not None:
        if report.failed:
            try:
                mutation_transaction_report = mutation_transaction.rollback("pipeline_failure")
                context["model_modified"] = False
            except Exception as exc:  # noqa: BLE001 - rollback failure is part of the final report.
                report.failed.append(
                    {
                        "id": "tuning.mutation_transaction",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        else:
            try:
                mutation_transaction_report = mutation_transaction.commit()
            except Exception as exc:  # noqa: BLE001 - commit failure must fail closed and roll back.
                report.failed.append(
                    {
                        "id": "tuning.mutation_transaction",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                mutation_transaction_report = mutation_transaction.rollback("commit_failure")
                context["model_modified"] = False

    return {
        "success": not report.failed,
        "dry_run": dry_run,
        "executed": report.executed,
        "skipped": report.skipped,
        "unavailable": [
            toggle_id for toggle_id, is_enabled in resolved.items()
            if is_enabled and not TOGGLES[toggle_id].get("available", False)
        ],
        "failed": report.failed,
        "model_summary": adapter.handle.summary(),
        "model_modified": bool(context.get("model_modified", False)),
        "quantization": context.get("quantization_report"),
        "architecture": context.get("architecture_report"),
        "provenance": context.get("provenance_report"),
        "reproducibility": context.get("reproducibility_report"),
        "memory": context.get("memory_report"),
        "weight_fingerprint": context.get("weight_fingerprint"),
        "router_collapse": context.get("router_collapse_report"),
        "multi_objective": context.get("multi_objective_report"),
        "security_scan": context.get("security_scan_report"),
        "baseline": context.get("baseline_report"),
        "benchmark_regression": context.get("benchmark_regression_report"),
        "version_compatibility": context.get("version_compatibility_report"),
        "tokenization_consistency": context.get("tokenization_consistency_report"),
        "capability_battery": context.get("capability_battery_report"),
        "latency": context.get("latency_report"),
        "throughput": context.get("throughput_report"),
        "resource_budget": context.get("resource_budget_report"),
        "kv_cache": context.get("kv_cache_report"),
        "audit": context.get("audit_report"),
        "topology_proposal": context.get("topology_proposal_report"),
        "mutation_transaction": mutation_transaction_report,
        "output_dir": context.get("exported_path"),
        "export": context.get("export_report"),
    }
