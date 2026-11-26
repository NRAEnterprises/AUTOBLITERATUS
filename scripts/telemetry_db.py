#!/usr/bin/env python3
"""Initialize and populate the opt-in AUTOBLITERATUS telemetry database."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from autobliteratus.observability.query_pack import (
    measurement_summary,
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

    args = parser.parse_args()

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
