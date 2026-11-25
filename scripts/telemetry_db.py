#!/usr/bin/env python3
"""Initialize and populate the opt-in AUTOBLITERATUS telemetry database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autobliteratus.observability.telemetry_store import (
    import_benchmark_report,
    import_resource_budget_report,
    import_results_csv,
    initialize_database,
    summarize_database,
)


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
    subparsers.add_parser("summary", help="print aggregate run and test counts")
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
        print(json.dumps(result.__dict__, sort_keys=True))
    else:
        print(json.dumps(summarize_database(db_path), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
