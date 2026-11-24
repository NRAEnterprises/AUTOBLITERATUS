"""Opt-in local telemetry warehouse for AUTOBLITERATUS test evidence.

The first increment imports the existing aggregate CSV ledger. It does not
capture live test output or enable any runtime pipeline stage; those integrations
must use the versioned schema and redacted event contract documented separately.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema-v1.sql"
_REQUIRED_COLUMNS = {
    "date",
    "commit",
    "platform",
    "command",
    "passed",
    "failed",
    "skipped",
    "notes",
}
_SECRET_KEY = (
    r"api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|token|"
    r"password|passwd|secret|client[_-]?secret|private[_-]?key|"
    r"secret[_-]?access[_-]?key"
)
_SECRET_IDENTIFIER = rf"(?:[A-Za-z][A-Za-z0-9]*[_-])*(?:{_SECRET_KEY})"
_SECRET_VALUE = r'''(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,;}&]+)'''
_AUTHORIZATION_HEADER = re.compile(
    rf'''(?ix)(?P<prefix>(?<![A-Za-z0-9_])["']?(?:[A-Za-z][A-Za-z0-9]*[_-])*authorization["']?\s*[:=]\s*)'''
    rf'''(?P<value>(?:Bearer|Basic)\s+[^\s,;]+|{_SECRET_VALUE})'''
)
_SECRET_ASSIGNMENT = re.compile(
    rf'''(?ix)(?P<prefix>(?<![A-Za-z0-9_])["']?(?:--)?(?:{_SECRET_IDENTIFIER})["']?\s*[:=]\s*)'''
    rf'''(?P<scheme>(?:Bearer|Basic)\s+)?(?P<value>{_SECRET_VALUE})'''
)
_CLI_SECRET_ARGUMENT = re.compile(
    rf'''(?ix)(?P<prefix>(?<![A-Za-z0-9_])--(?:{_SECRET_IDENTIFIER})\s+)'''
    rf'''(?P<scheme>(?:Bearer|Basic)\s+)?(?P<value>{_SECRET_VALUE})'''
)
_BEARER_TOKEN = re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]+")
_PROVIDER_TOKEN = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+|"
    r"glpat-[A-Za-z0-9_-]+|xox[baprs]-[A-Za-z0-9-]+|ya29\.[A-Za-z0-9_-]+)"
)


@dataclass(frozen=True)
class ImportResult:
    rows_seen: int
    rows_inserted: int
    source_sha256: str


def redact_text(value: str) -> str:
    """Redact common credential forms before storing free-text telemetry."""

    def replace_assignment(match: re.Match[str]) -> str:
        secret = match.group("value")
        if secret[:1] in ("'", '"') and secret[-1:] == secret[:1]:
            quote = secret[0]
            replacement = f"{quote}[REDACTED]{quote}"
        else:
            replacement = "[REDACTED]"
        return match.group("prefix") + (match.groupdict().get("scheme") or "") + replacement

    redacted = _AUTHORIZATION_HEADER.sub(
        lambda match: match.group("prefix") + "[REDACTED]", value
    )
    redacted = _SECRET_ASSIGNMENT.sub(replace_assignment, redacted)
    redacted = _CLI_SECRET_ARGUMENT.sub(replace_assignment, redacted)
    redacted = _BEARER_TOKEN.sub(r"\1[REDACTED]", redacted)
    return _PROVIDER_TOKEN.sub("[REDACTED]", redacted)


def _duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - exercised without optional extra
        raise RuntimeError(
            "DuckDB telemetry support is optional. Install DuckDB with "
            "`pip install 'duckdb>=2.0,<3.0'`; in a fully provisioned project "
            "environment, the `observability` extra may be installed instead."
        ) from exc
    return duckdb


def initialize_database(database_path: str | Path) -> Path:
    """Create the local database and apply the checked-in v1 schema."""
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not _SCHEMA_PATH.is_file():
        raise FileNotFoundError(f"Telemetry schema not found: {_SCHEMA_PATH}")
    connection = _duckdb().connect(str(path))
    try:
        connection.execute(_SCHEMA_PATH.read_text(encoding="utf-8"))
    finally:
        connection.close()
    return path


def _row_run_id(row: dict[str, str]) -> str:
    canonical = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(("autobliteratus-csv-summary-v1\0" + canonical).encode("utf-8")).hexdigest()


def import_results_csv(csv_path: str | Path, database_path: str | Path) -> ImportResult:
    """Import legacy aggregate test rows, deduplicating by canonical row hash."""
    source_path = Path(csv_path)
    raw = source_path.read_bytes()
    source_sha256 = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fields = set(reader.fieldnames or [])
    missing = _REQUIRED_COLUMNS - fields
    if missing:
        raise ValueError(f"Test ledger is missing required columns: {', '.join(sorted(missing))}")

    path = initialize_database(database_path)
    connection = _duckdb().connect(str(path))
    rows_seen = 0
    rows_inserted = 0
    connection.execute("BEGIN TRANSACTION")
    try:
        for source_row in reader:
            rows_seen += 1
            row = {key: (value or "") for key, value in source_row.items()}
            try:
                run_date = date.fromisoformat(row["date"])
                passed, failed, skipped = (int(row[key]) for key in ("passed", "failed", "skipped"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid date or test count in CSV row {rows_seen + 1}") from exc
            if min(passed, failed, skipped) < 0:
                raise ValueError(f"Negative test count in CSV row {rows_seen + 1}")

            status = (
                "failed"
                if failed
                else "passed"
                if passed and not skipped
                else "partial"
                if passed or skipped
                else "no_results"
            )
            run_id = _row_run_id(row)
            result = connection.execute(
                """INSERT INTO pipeline_runs
                   (run_id, run_date, commit_sha, platform, command, status,
                    source_name, source_sha256, notes_redacted)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (run_id) DO NOTHING
                   RETURNING run_id""",
                [
                    run_id,
                    run_date,
                    row["commit"],
                    redact_text(row["platform"]),
                    redact_text(row["command"]),
                    status,
                    source_path.name,
                    source_sha256,
                    redact_text(row["notes"]),
                ],
            ).fetchone()
            if result is None:
                continue
            rows_inserted += 1
            connection.execute(
                """INSERT INTO test_summaries (run_id, passed, failed, skipped, total)
                   VALUES (?, ?, ?, ?, ?)""",
                [run_id, passed, failed, skipped, passed + failed + skipped],
            )
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return ImportResult(rows_seen, rows_inserted, source_sha256)


def summarize_database(database_path: str | Path) -> dict[str, int]:
    """Return aggregate counts from the local telemetry database."""
    connection = _duckdb().connect(str(database_path), read_only=True)
    try:
        runs = connection.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
        tests = connection.execute(
            "SELECT COALESCE(SUM(total), 0), COALESCE(SUM(failed), 0), "
            "COALESCE(SUM(skipped), 0) FROM test_summaries"
        ).fetchone()
        return {
            "runs": int(runs),
            "tests": int(tests[0]),
            "failed": int(tests[1]),
            "skipped": int(tests[2]),
        }
    finally:
        connection.close()
