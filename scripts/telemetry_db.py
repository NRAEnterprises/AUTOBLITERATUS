#!/usr/bin/env python3
"""Initialize and populate the opt-in AUTOBLITERATUS telemetry database."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

from autobliteratus.observability.artifact_manifest import import_artifact_manifest
from autobliteratus.observability.quality_outcomes import import_quality_outcomes
from autobliteratus.observability.query_pack import (
    measurement_summary,
    quality_latency_pareto_report,
    repeated_test_failures,
    run_outcomes,
    token_usage_summary,
)
from autobliteratus.observability.stage_events import import_stage_events
from autobliteratus.observability.telemetry_store import (
    import_benchmark_report,
    import_resource_budget_report,
    import_results_csv,
    initialize_database,
    summarize_database,
)


def _date_argument(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD format") from exc
    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD format")
    return parsed


def _json_object_argument(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("dimensions must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("dimensions must be a JSON object")
    return parsed


def _dimension_keys_argument(value: str) -> tuple[str, ...]:
    keys = tuple(part.strip() for part in value.split(","))
    if not keys or any(not key for key in keys):
        raise argparse.ArgumentTypeError("dimension keys must be a non-empty comma-separated list")
    if len(set(keys)) != len(keys):
        raise argparse.ArgumentTypeError("dimension keys must not contain duplicates")
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in keys):
        raise argparse.ArgumentTypeError("dimension keys must be simple identifier strings")
    return keys


def _positive_integer_argument(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _add_date_limit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start-date", type=_date_argument, help="inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=_date_argument, help="inclusive end date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=200, help="maximum report rows (hard-capped at 1000)")


def _print_json(value: object) -> None:
    print(json.dumps(value, sort_keys=True, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default="testing/autobliteratus_telemetry.duckdb",
        help="DuckDB path (default: testing/autobliteratus_telemetry.duckdb)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="create the database and apply schema v1")
    importer = subparsers.add_parser("import-results", help="import legacy testing/results.csv")
    importer.add_argument("csv_path", nargs="?", default="testing/results.csv")
    resource_importer = subparsers.add_parser(
        "import-resource-budget",
        help="import a report-only resource-budget v1 JSON report",
    )
    resource_importer.add_argument("report_path", help="path to resource-budget report JSON")
    benchmark_importer = subparsers.add_parser(
        "import-benchmark",
        help="import supported telemetry benchmark JSON or versioned long-form CSV",
    )
    benchmark_importer.add_argument("report_path", help="path to benchmark .json or .csv report")
    stage_importer = subparsers.add_parser(
        "import-stage-events",
        help="import bounded, versioned stage-event JSONL",
    )
    stage_importer.add_argument("jsonl_path", help="path to stage-event JSONL")
    quality_importer = subparsers.add_parser(
        "import-quality-outcomes",
        help="import a strict numeric quality-outcomes v1 JSON report",
    )
    quality_importer.add_argument("report_path", help="path to quality-outcomes JSON")
    artifact_importer = subparsers.add_parser(
        "import-artifact-manifest",
        help="import asserted artifact-manifest v1 metadata",
    )
    artifact_importer.add_argument("manifest_path", help="path to artifact-manifest JSON")
    subparsers.add_parser("summary", help="print aggregate run and test counts")

    outcomes = subparsers.add_parser("run-outcomes", help="report run outcomes by commit, platform, and status")
    outcomes.add_argument("--commit-sha")
    outcomes.add_argument("--platform")
    outcomes.add_argument("--status")
    _add_date_limit_arguments(outcomes)

    failures = subparsers.add_parser(
        "repeated-test-failures", help="report repeated test-failure fingerprints"
    )
    failures.add_argument("--test-id")
    failures.add_argument("--failure-fingerprint")
    failures.add_argument("--commit-sha")
    failures.add_argument("--platform")
    failures.add_argument("--min-runs", type=int, default=2)
    _add_date_limit_arguments(failures)

    measurements = subparsers.add_parser(
        "measurement-summary", help="summarize normalized measurements"
    )
    measurements.add_argument("--metric")
    measurements.add_argument("--stage")
    measurements.add_argument(
        "--dimensions", type=_json_object_argument, metavar="JSON",
        help='dimension filters as a JSON object, for example {"backend":"cpu"}',
    )
    _add_date_limit_arguments(measurements)

    usage = subparsers.add_parser("token-usage-summary", help="summarize recorded LLM token usage")
    usage.add_argument("--stage")
    usage.add_argument("--provider")
    usage.add_argument("--model")
    usage.add_argument("--purpose")
    _add_date_limit_arguments(usage)

    pareto = subparsers.add_parser(
        "quality-latency-pareto",
        help="compare quality and latency and identify Pareto-efficient candidates",
    )
    pareto.add_argument("--quality-metric", required=True)
    pareto.add_argument("--latency-metric", required=True)
    pareto.add_argument("--cohort-by", required=True, type=_dimension_keys_argument)
    pareto.add_argument("--candidate-by", required=True, type=_dimension_keys_argument)
    pareto.add_argument("--quality-stage")
    pareto.add_argument("--latency-stage")
    pareto.add_argument(
        "--quality-direction", choices=("higher", "lower"), default="higher"
    )
    pareto.add_argument("--start-date", type=_date_argument)
    pareto.add_argument("--end-date", type=_date_argument)
    pareto.add_argument("--min-paired-runs", type=_positive_integer_argument, default=1)
    pareto.add_argument("--limit", type=_positive_integer_argument, default=200)

    args = parser.parse_args()

    if args.command == "quality-latency-pareto" and (
        set(args.cohort_by) & set(args.candidate_by)
    ):
        parser.error("--cohort-by and --candidate-by dimensions must be disjoint")

    db_path = Path(args.db)
    if args.command == "init":
        initialize_database(db_path)
        print(f"Initialized telemetry database: {db_path}")
    elif args.command == "import-results":
        result = import_results_csv(args.csv_path, db_path)
        print(json.dumps(result.__dict__, sort_keys=True))
    elif args.command == "import-resource-budget":
        result = import_resource_budget_report(args.report_path, db_path)
        print(json.dumps(result.__dict__, sort_keys=True))
    elif args.command == "import-benchmark":
        result = import_benchmark_report(args.report_path, db_path)
        _print_json(result.__dict__)
    elif args.command == "import-stage-events":
        result = import_stage_events(args.jsonl_path, db_path)
        _print_json(result.__dict__)
    elif args.command == "import-quality-outcomes":
        result = import_quality_outcomes(args.report_path, db_path)
        _print_json(result.__dict__)
    elif args.command == "import-artifact-manifest":
        result = import_artifact_manifest(args.manifest_path, db_path)
        _print_json(result.__dict__)
    elif args.command == "summary":
        _print_json(summarize_database(db_path))
    elif args.command == "run-outcomes":
        report = run_outcomes(
            db_path,
            commit_sha=args.commit_sha,
            platform=args.platform,
            status=args.status,
            start_date=args.start_date,
            end_date=args.end_date,
            limit=args.limit,
        )
        _print_json({"name": report.name, "rows": report.rows, "metadata": report.metadata})
    elif args.command == "repeated-test-failures":
        report = repeated_test_failures(
            db_path,
            test_id=args.test_id,
            failure_fingerprint=args.failure_fingerprint,
            commit_sha=args.commit_sha,
            platform=args.platform,
            start_date=args.start_date,
            end_date=args.end_date,
            min_runs=args.min_runs,
            limit=args.limit,
        )
        _print_json({"name": report.name, "rows": report.rows, "metadata": report.metadata})
    elif args.command == "measurement-summary":
        report = measurement_summary(
            db_path,
            metric=args.metric,
            stage=args.stage,
            dimensions=args.dimensions,
            start_date=args.start_date,
            end_date=args.end_date,
            limit=args.limit,
        )
        _print_json({"name": report.name, "rows": report.rows, "metadata": report.metadata})
    elif args.command == "quality-latency-pareto":
        report = quality_latency_pareto_report(
            db_path,
            quality_metric=args.quality_metric,
            latency_metric=args.latency_metric,
            cohort_dimensions=args.cohort_by,
            candidate_dimensions=args.candidate_by,
            quality_stage=args.quality_stage,
            latency_stage=args.latency_stage,
            quality_direction=args.quality_direction,
            start_date=args.start_date,
            end_date=args.end_date,
            min_paired_runs=args.min_paired_runs,
            limit=args.limit,
        )
        _print_json(report.as_dict())
    else:
        report = token_usage_summary(
            db_path,
            stage=args.stage,
            provider=args.provider,
            model=args.model,
            purpose=args.purpose,
            start_date=args.start_date,
            end_date=args.end_date,
            limit=args.limit,
        )
        _print_json({"name": report.name, "rows": report.rows, "metadata": report.metadata})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
