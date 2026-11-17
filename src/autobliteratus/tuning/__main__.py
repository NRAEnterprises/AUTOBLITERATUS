"""Module entry point. Enables ``python -m autobliteratus``."""

from __future__ import annotations

import json
import sys


def main(argv: list[str] | None = None) -> int:
    from autobliteratus.cli import parse_args
    from autobliteratus.pipeline import run_pipeline

    args = parse_args(argv)
    if not args.model_path:
        print("--model-path is required for pipeline execution")
        return 1

    result = run_pipeline(
        model_path=args.model_path,
        work_dir=args.work_dir,
        output_dir=args.output_dir,
        quantization=args.quantization,
        target_frameworks=args.target_frameworks,
        deployment_target=args.deployment_target,
        enable=args.enable,
        disable=args.disable,
        allow_mutation=getattr(args, "allow_mutation", False),
        dry_run=args.dry_run,
        ablation_strength=args.ablation_strength,
        ablation_component=args.ablation_component,
        trust_remote_code=args.trust_remote_code,
        calibration_data=args.calibration_data,
        calibration_samples=args.calibration_samples,
        calibration_max_length=args.calibration_max_length,
        calibration_momentum=args.calibration_momentum,
        calibration_streamline=args.calibration_streamline,
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
        max_memory=args.max_memory,
        offload_folder=args.offload_folder,
        max_shard_size=args.max_shard_size,
        plan_only=args.plan_only,
        stage_options=getattr(args, "stage_options", None),
    )
    if result.get("plan_only"):
        print(json.dumps(result["plan"], indent=2, ensure_ascii=False))
        return 0 if result["success"] else 1
    print(f"executed: {len(result['executed'])}")
    print(f"skipped:  {len(result['skipped'])}")
    print(f"unavailable: {len(result.get('unavailable', []))}")
    print(f"failed:   {len(result['failed'])}")
    print(f"model:    {result.get('model_summary', '')}")
    print(f"modified: {result.get('model_modified', False)}")
    if result.get("quantization"):
        print(f"quantization: {result['quantization'].get('mode', '')}")
    router_report = result.get("router_collapse")
    if router_report:
        print(
            "router diagnostics: "
            f"{router_report.get('status')} "
            f"(collapsed layers: {len(router_report.get('collapsed_layers', []))})"
        )
    objective_report = result.get("multi_objective")
    if objective_report:
        selected = objective_report.get("selected_candidate") or {}
        print(
            "multi-objective: "
            f"{objective_report.get('status')} "
            f"(candidates: {objective_report.get('candidate_count', 0)}, "
            f"frontier: {len(objective_report.get('pareto_frontier', []))}, "
            f"selected: {selected.get('candidate_id', 'none')})"
        )
    security_report = result.get("security_scan")
    if security_report:
        print(
            "numeric security scan: "
            f"{security_report.get('status')} "
            f"(parameters scanned: {security_report.get('parameters_scanned', 0)}, "
            f"findings: {len(security_report.get('findings', []))})"
        )
    baseline_report = result.get("baseline")
    if baseline_report:
        print(
            "evaluation baseline: "
            f"{baseline_report.get('status')} "
            f"(samples: {baseline_report.get('sample_count', 0)}, "
            f"tokens: {baseline_report.get('scored_tokens', 0)})"
        )
    regression_report = result.get("benchmark_regression")
    if regression_report:
        print(
            "benchmark regression gate: "
            f"{regression_report.get('status')} "
            f"(regressions: {len(regression_report.get('regressions', []))})"
        )
    audit_report = result.get("audit")
    if audit_report:
        print(f"audit: {audit_report.get('status')} ({audit_report.get('markdown_path', '')})")
    if result.get("output_dir"):
        print(f"output:   {result['output_dir']}")
    for failure in result.get("failed", []):
        print(f"stage error [{failure['id']}]: {failure['error']}")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
