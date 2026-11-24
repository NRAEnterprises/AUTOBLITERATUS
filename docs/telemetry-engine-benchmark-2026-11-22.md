# Telemetry ledger engine comparison — 2026-11-22

## Result

The final rerun passed exact-result parity checks for CSV/Python, SQLite, and DuckDB on the five actual rows in `testing/results.csv` (3,167 bytes; source SHA-256 `06035522e1161003ef39f626ebb5f3a76e0524876229f39b0bc62d90a983fafd`). No performance-workload rows were synthesized. Each of the three query outputs had the same canonical SHA-256 hash across engines, and parity was checked on all 1,000 timed executions and 25 warmups per query and engine.

For this tiny workload, SQLite had the lowest repeated query medians, CSV/Python was next, and DuckDB was slowest. The medians were 0.0085–0.0112 ms for SQLite, 0.0298–0.0301 ms for CSV/Python, and 0.7039–0.8741 ms for DuckDB. These are host- and workload-specific observations, not general engine rankings or scaling evidence.

## Measurements

| Engine | Build/import time (ms) | File/data bytes | Summary median / p95 (ms) | Platform-group median / p95 (ms) | Filtered-run median / p95 (ms) |
|---|---:|---:|---:|---:|---:|
| CSV/Python | n/a | 3,167 | 0.0298 / 0.0348 | 0.0298 / 0.0411 | 0.0301 / 0.0432 |
| SQLite 3.45.1 | 4.174 | 20,480 | 0.0085 / 0.0091 | 0.0101 / 0.0132 | 0.0112 / 0.0114 |
| DuckDB 2.0.1 | 79.130 | 2,109,440 | 0.7039 / 0.7975 | 0.8541 / 0.9741 | 0.8741 / 0.9624 |

Database build/import time includes CSV parsing, schema/table setup, and insertion. CSV/Python reopens and parses the source CSV for each query and has no separate import phase. SQLite used a file-backed normalized `pipeline_runs`/`test_summaries` schema, default `DELETE` journal mode, no FTS5 table, and one reused connection. The tested SQL is structured aggregation and filtering, not full-text search. DuckDB used the project's v1 schema and importer, one reused read-only connection, and one query thread. The database schemas and catalog footprints differ, so file size and setup time are operational context, not a same-schema storage-efficiency comparison.

The queries were: aggregate run and test totals; group runs and outcome totals by platform; and return sorted run details where `passed >= 35`. There were 1,000 timed runs plus 25 warmups per query/engine. Every output was checked against CSV/Python on every run. Latency uses `perf_counter_ns`. The JSON includes environment details, output hashes, per-query distributions, and repetition counts.

Each engine/query path also executes an untimed parity query before timed measurement. The field `first_measured_after_parity_ms` is therefore not a cold-cache or first-use measurement. OS and database caches were not cleared.

## Interpretation and limits

This comparison does not meet the project's proposed representative-workload gate. The source has only five run rows and 173 aggregate test outcomes. The observed SQLite latency lead is specific to the selected small local aggregation/filter queries. It says nothing about larger histories, Parquet scans, concurrency, full-text search, write throughput, crash recovery, or other query patterns.

No token-efficiency test was performed because this benchmark does not invoke an LLM or measure tokens per correct answer. No engine superiority or token-savings claim is warranted. Treat this as a parity-checked small-ledger baseline. Accumulate authentic pipeline telemetry, define representative multi-dimensional queries, and rerun the same correctness-first harness before making a promotion decision.

## Reproduction and artifacts

From the repository root, with DuckDB installed in the selected Python environment:

```bash
PYTHONPATH=src python scripts/benchmark_telemetry_engines.py \
  --csv testing/results.csv \
  --repetitions 1000 \
  --warmups 25 \
  --output benchmarks/results/telemetry-ledger-engine-benchmark-2026-11-22.json
```

The [machine-readable measurements and parity hashes](../benchmarks/results/telemetry-ledger-engine-benchmark-2026-11-22.json) include versions and environment details. The [reproducible benchmark script](../scripts/benchmark_telemetry_engines.py) builds temporary database copies and deletes them after measurement; it does not alter the local pilot database. See the separate [system assessment](telemetry-system-assessment-2026-11-22.md) for importer, privacy, resilience, and read-concurrency checks.
