from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("duckdb", reason="DuckDB is an optional observability extra")


def test_opt_in_pytest_plugin_records_redacted_per_test_results(tmp_path):
    project_root = os.path.dirname(os.path.dirname(__file__))
    test_file = tmp_path / "test_sample_private_parameter.py"
    database = tmp_path / "telemetry.duckdb"
    test_file.write_text(
        """import pytest


def test_passed():
    assert 2 + 2 == 4


@pytest.mark.parametrize('credential', ['api_key=example-secret-value'])
def test_failure_fingerprint_only(credential):
    assert False, credential


def test_skipped():
    pytest.skip('skip details must not be persisted')
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(project_root, "src")
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "autobliteratus.observability.pytest_plugin",
            "--autobliteratus-db",
            str(database),
            str(test_file),
        ],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert process.returncode == 1, process.stdout + process.stderr

    import duckdb

    connection = duckdb.connect(str(database), read_only=True)
    try:
        outcomes = connection.execute(
            "SELECT outcome, COUNT(*) FROM test_results GROUP BY outcome ORDER BY outcome"
        ).fetchall()
        names = connection.execute("SELECT test_id FROM test_results").fetchall()
        combined = " ".join(
            str(row)
            for table in ("pipeline_runs", "test_results", "measurements")
            for row in connection.execute(f"SELECT * FROM {table}").fetchall()
        )
        summary = connection.execute("SELECT passed, failed, skipped, total FROM test_summaries").fetchone()
    finally:
        connection.close()

    assert dict(outcomes) == {"failed": 1, "passed": 1, "skipped": 1}
    assert summary == (1, 1, 1, 3)
    assert len(names) == 3
    assert all(len(row[0]) == 64 for row in names)
    assert "example-secret-value" not in combined
    assert "test_failure_fingerprint_only" not in combined
    assert "skip details must not be persisted" not in combined
