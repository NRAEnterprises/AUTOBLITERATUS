from __future__ import annotations

import json

import duckdb
import pytest

from autobliteratus.observability.telemetry_store import import_resource_budget_report


def _report() -> dict:
    return {
        "stage_id": "tuning.resource_budget_gate",
        "schema_version": 1,
        "status": "violations",
        "budget": {
            "max_latency_ms": 250.0,
            "min_tokens_per_second": 40.0,
        },
        "observations": [
            {
                "candidate_index": 0,
                "batch_size": 4,
                "sequence_length": 512,
                "tokens_per_second": 38.2,
                "latency_ms": 291.0,
                "host_rss_bytes": 2_147_483_648,
                "accelerator_allocated_bytes": None,
            }
        ],
        "candidate_outcomes": [
            {
                "candidate_index": 1,
                "status": "oom",
                "batch_size": 8,
                "sequence_length": 2048,
                "failure_metadata": {
                    "kind": "out_of_memory",
                    "backend": "cuda:0",
                    "reason": "api_key=must-not-be-stored",
                    "cleanup": {
                        "attempted": True,
                        "succeeded": False,
                        "action": "cuda.empty_cache",
                    },
                },
            }
        ],
        "violations": [
            {
                "observation_index": 0,
                "candidate_index": 0,
                "metric": "latency_ms",
                "budget": "max_latency_ms",
                "observed": 291.0,
                "limit": 250.0,
                "operator": "<=",
                "unit": "milliseconds",
                "batch_size": 4,
                "sequence_length": 512,
            },
            {
                "observation_index": 0,
                "candidate_index": 0,
                "metric": "tokens_per_second",
                "budget": "min_tokens_per_second",
                "observed": 38.2,
                "limit": 40.0,
                "operator": ">=",
                "unit": "tokens_per_second",
                "batch_size": 4,
                "sequence_length": 512,
            },
        ],
        "missing_metrics": [],
        "optimization_performed": False,
        "model_modified": False,
        "artifacts": [],
        "warnings": ["debug token=warning-secret"],
        "unrecognized_extension": {"client_secret": "extension-secret"},
    }


def _write_report(path, report: dict) -> None:
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")


def test_import_resource_budget_report_normalizes_and_replays_idempotently(tmp_path):
    report_path = tmp_path / "budget.json"
    database = tmp_path / "telemetry.duckdb"
    _write_report(report_path, _report())

    first = import_resource_budget_report(report_path, database)
    second = import_resource_budget_report(report_path, database)

    assert first.inserted is True
    assert second.inserted is False
    assert first.report_id == second.report_id
    assert first.observations_seen == 1
    assert first.findings_seen == 3

    connection = duckdb.connect(str(database), read_only=True)
    try:
        report_row = connection.execute(
            "SELECT status, source_sha256, observation_count, finding_count, budget "
            "FROM resource_budget_reports"
        ).fetchone()
        observation = connection.execute(
            "SELECT candidate_index, batch_size, sequence_length, tokens_per_second, "
            "latency_ms, host_rss_bytes, accelerator_allocated_bytes "
            "FROM resource_budget_observations"
        ).fetchone()
        findings = connection.execute(
            "SELECT finding_kind, code, metric, budget_key, observed, limit_value, details "
            "FROM resource_budget_findings ORDER BY finding_kind, finding_index"
        ).fetchall()
    finally:
        connection.close()

    assert report_row[0] == "violations"
    assert report_row[2:4] == (1, 3)
    assert observation == (0, 4, 512, 38.2, 291.0, 2_147_483_648, None)
    assert {row[0] for row in findings} == {"candidate_outcome", "violation"}
    persisted = json.dumps((report_row, findings), default=str)
    assert "cuda:0" in persisted
    assert "cuda.empty_cache" in persisted
    assert "warning-secret" not in persisted
    assert "must-not-be-stored" not in persisted
    assert "extension-secret" not in persisted


def test_import_rejects_false_or_inconsistent_reports_before_database_write(tmp_path):
    report = _report()
    report["schema_version"] = True
    path = tmp_path / "boolean-version.json"
    _write_report(path, report)
    with pytest.raises(ValueError, match="schema version"):
        import_resource_budget_report(path, tmp_path / "telemetry.duckdb")

    report = _report()
    report["status"] = "within_budget"
    path = tmp_path / "status-mismatch.json"
    _write_report(path, report)
    with pytest.raises(ValueError, match="status conflicts"):
        import_resource_budget_report(path, tmp_path / "telemetry.duckdb")

    assert not (tmp_path / "telemetry.duckdb").exists()


def test_import_rejects_violation_that_does_not_match_source_observation(tmp_path):
    report = _report()
    report["violations"][0]["observed"] = 249.0
    path = tmp_path / "bad-violation.json"
    _write_report(path, report)

    with pytest.raises(ValueError, match="differs from its observation"):
        import_resource_budget_report(path, tmp_path / "telemetry.duckdb")


def test_import_rejects_duplicate_keys_and_non_finite_json(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"status":"within_budget","status":"violations"}', encoding="utf-8")
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        import_resource_budget_report(duplicate, tmp_path / "duplicate.duckdb")

    non_finite = tmp_path / "non-finite.json"
    non_finite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        import_resource_budget_report(non_finite, tmp_path / "non-finite.duckdb")


def test_import_rejects_oversized_report(tmp_path):
    report_path = tmp_path / "large.json"
    report_path.write_bytes(b" " * (4 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="4 MiB"):
        import_resource_budget_report(report_path, tmp_path / "telemetry.duckdb")


def test_import_rejects_omitted_budget_violation(tmp_path):
    report = _report()
    report["violations"].pop()
    path = tmp_path / "omitted-violation.json"
    _write_report(path, report)

    with pytest.raises(ValueError, match="violations are incomplete"):
        import_resource_budget_report(path, tmp_path / "telemetry.duckdb")


def test_import_rejects_unreported_unavailable_budgeted_metric(tmp_path):
    report = _report()
    report["budget"]["max_accelerator_allocated_bytes"] = 3_000_000_000
    path = tmp_path / "omitted-missing-metric.json"
    _write_report(path, report)

    with pytest.raises(ValueError, match="missing metrics are incomplete"):
        import_resource_budget_report(path, tmp_path / "telemetry.duckdb")


def test_import_rejects_unrecognized_candidate_status_text(tmp_path):
    report = _report()
    report["candidate_outcomes"][0]["status"] = "password=do-not-store"
    path = tmp_path / "unsafe-status.json"
    _write_report(path, report)

    with pytest.raises(ValueError, match="status is unsupported"):
        import_resource_budget_report(path, tmp_path / "telemetry.duckdb")


def test_import_converts_deep_json_recursion_to_validation_error(tmp_path):
    encoded = json.dumps(_report())
    deeply_nested = encoded[:-1] + ',"ignored_extension":' + "[" * 2000 + "0" + "]" * 2000 + "}"
    path = tmp_path / "deep.json"
    path.write_text(deeply_nested, encoding="utf-8")

    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        import_resource_budget_report(path, tmp_path / "telemetry.duckdb")


def test_import_preserves_large_integer_byte_violation_exactly(tmp_path):
    report = _report()
    limit = 9_007_199_254_740_992
    observed = 9_007_199_254_740_993
    report["status"] = "violations"
    report["budget"] = {"max_host_rss_bytes": limit}
    report["observations"] = [
        {
            "candidate_index": 0,
            "batch_size": 1,
            "sequence_length": 64,
            "tokens_per_second": 10.0,
            "latency_ms": 1.0,
            "host_rss_bytes": observed,
            "accelerator_allocated_bytes": None,
        }
    ]
    report["candidate_outcomes"] = []
    report["violations"] = [
        {
            "observation_index": 0,
            "candidate_index": 0,
            "metric": "host_rss_bytes",
            "budget": "max_host_rss_bytes",
            "observed": observed,
            "limit": limit,
            "operator": "<=",
            "unit": "bytes",
            "batch_size": 1,
            "sequence_length": 64,
        }
    ]
    report["warnings"] = []
    path = tmp_path / "large-bytes.json"
    _write_report(path, report)
    database = tmp_path / "telemetry.duckdb"

    import_resource_budget_report(path, database)
    connection = duckdb.connect(str(database), read_only=True)
    try:
        saved_observed, saved_limit = connection.execute(
            "SELECT CAST(observed AS VARCHAR), CAST(limit_value AS VARCHAR) "
            "FROM resource_budget_findings WHERE finding_kind = 'violation'"
        ).fetchone()
    finally:
        connection.close()

    assert int(saved_observed) == observed
    assert int(saved_limit) == limit


def test_import_reports_numeric_and_duckdb_integer_overflow_as_validation_errors(tmp_path):
    report = _report()
    report["budget"]["max_host_rss_bytes"] = 10**400
    path = tmp_path / "large-budget.json"
    _write_report(path, report)
    import_resource_budget_report(path, tmp_path / "large-budget.duckdb")

    report = _report()
    report["observations"][0]["tokens_per_second"] = 10**400
    path = tmp_path / "unrepresentable-double.json"
    _write_report(path, report)
    with pytest.raises(ValueError, match="out-of-range tokens_per_second"):
        import_resource_budget_report(path, tmp_path / "overflow.duckdb")

    report = _report()
    report["observations"][0]["candidate_index"] = 2**31
    path = tmp_path / "oversized-index.json"
    _write_report(path, report)
    with pytest.raises(ValueError, match="candidate_index"):
        import_resource_budget_report(path, tmp_path / "index.duckdb")

    report = _report()
    report["observations"][0]["host_rss_bytes"] = 2**63
    path = tmp_path / "oversized-bytes.json"
    _write_report(path, report)
    with pytest.raises(ValueError, match="host_rss_bytes"):
        import_resource_budget_report(path, tmp_path / "bytes.duckdb")
