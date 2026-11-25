from __future__ import annotations

import csv

import pytest

pytest.importorskip("duckdb", reason="DuckDB is an optional observability extra")

from autobliteratus.observability.telemetry_store import (
    import_results_csv,
    initialize_database,
    redact_text,
    summarize_database,
)


def _write_ledger(path, rows):
    fields = ["date", "commit", "platform", "command", "passed", "failed", "skipped", "notes"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_initialize_database_creates_schema(tmp_path):
    path = tmp_path / "telemetry.duckdb"

    assert initialize_database(path) == path
    assert path.exists()


def test_csv_import_is_idempotent_and_redacts_secrets(tmp_path):
    source = tmp_path / "results.csv"
    database = tmp_path / "telemetry.duckdb"
    _write_ledger(
        source,
        [
            {
                "date": "2026-11-22",
                "commit": "abc123",
                "platform": "linux-x86_64 token=platform-secret",
                "command": (
                    "pytest --token=command-secret Authorization: Bearer header-secret "
                    "ghp_BENCHMARK_FAKE_SECRET OPENAI_API_KEY=env-secret "
                    "GITHUB_TOKEN=github-env-secret"
                ),
                "passed": "7",
                "failed": "1",
                "skipped": "2",
                "notes": (
                    'api_key=secret-value; password="password with spaces"; '
                    'token=note-secret; {"client_secret": "json-secret"}; failure=assertion'
                ),
            }
        ],
    )

    first = import_results_csv(source, database)
    second = import_results_csv(source, database)

    assert first.rows_seen == 1
    assert first.rows_inserted == 1
    assert second.rows_seen == 1
    assert second.rows_inserted == 0
    assert first.source_sha256 == second.source_sha256
    assert summarize_database(database) == {"runs": 1, "tests": 10, "failed": 1, "skipped": 2}

    import duckdb

    connection = duckdb.connect(str(database), read_only=True)
    try:
        command, platform, note = connection.execute(
            "SELECT command, platform, notes_redacted FROM pipeline_runs"
        ).fetchone()
    finally:
        connection.close()
    persisted = f"{command} {platform} {note}"
    for canary in (
        "command-secret",
        "header-secret",
        "BENCHMARK_FAKE_SECRET",
        "env-secret",
        "github-env-secret",
        "platform-secret",
        "secret-value",
        "password with spaces",
        "note-secret",
        "json-secret",
    ):
        assert canary not in persisted
    assert persisted.count("[REDACTED]") >= 8


@pytest.mark.parametrize(
    ("sample", "canary"),
    [
        ("Authorization: Bearer auth-header-secret", "auth-header-secret"),
        ('"Authorization": "Basic basic-header-secret"', "basic-header-secret"),
        ("token=assignment-secret", "assignment-secret"),
        ("OPENAI_API_KEY=environment-secret", "environment-secret"),
        ("AWS_SECRET_ACCESS_KEY=aws-secret", "aws-secret"),
        ("--access-token cli-secret", "cli-secret"),
        ('{"client_secret": "json secret"}', "json secret"),
        (r'{"token": "prefix \" escaped-secret suffix"}', "escaped-secret"),
        ("Bearer standalone-secret", "standalone-secret"),
        ("ghp_BENCHMARK_FAKE_SECRET", "BENCHMARK_FAKE_SECRET"),
        ("github_pat_BENCHMARK_FAKE_SECRET", "BENCHMARK_FAKE_SECRET"),
    ],
)
def test_redact_text_covers_common_credential_shapes(sample, canary):
    assert canary not in redact_text(sample)


def test_import_rejects_bad_counts_without_silently_coercing(tmp_path):
    source = tmp_path / "results.csv"
    _write_ledger(
        source,
        [
            {
                "date": "2026-11-22",
                "commit": "valid-row",
                "platform": "linux-x86_64",
                "command": "pytest -q",
                "passed": "1",
                "failed": "0",
                "skipped": "0",
                "notes": "valid row must roll back",
            },
            {
                "date": "2026-11-22",
                "commit": "abc123",
                "platform": "linux-x86_64",
                "command": "pytest -q",
                "passed": "not-a-number",
                "failed": "0",
                "skipped": "0",
                "notes": "",
            }
        ],
    )

    with pytest.raises(ValueError, match="Invalid date or test count"):
        import_results_csv(source, tmp_path / "telemetry.duckdb")

    assert summarize_database(tmp_path / "telemetry.duckdb") == {
        "runs": 0,
        "tests": 0,
        "failed": 0,
        "skipped": 0,
    }


def test_import_rejects_oversized_legacy_csv_before_database_creation(tmp_path):
    source = tmp_path / "oversized-results.csv"
    with source.open("wb") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)

    with pytest.raises(ValueError, match="64 MiB"):
        import_results_csv(source, tmp_path / "telemetry.duckdb")

    assert not (tmp_path / "telemetry.duckdb").exists()


def test_initialize_database_restricts_local_file_permissions(tmp_path):
    import os
    import stat

    if os.name == "nt":
        pytest.skip("POSIX file mode bits are not portable to Windows")
    path = tmp_path / "private-telemetry.duckdb"
    initialize_database(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    path.chmod(0o644)
    initialize_database(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
