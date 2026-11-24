# AUTOBLITERATUS telemetry system assessment — 2026-11-22

## Overall result

The telemetry prototype passes several narrow correctness checks on the actual ledger, but it is not ready to support production-scale, privacy, or engine-superiority claims. The inspected `testing/results.csv` contains five run-summary rows and 3,167 bytes. Those rows represent 173 aggregate test outcomes, with zero failures and zero skips. No rows were added to the performance workloads. Every throughput and latency result below is a smoke-scale observation, not a capacity estimate.

Ratings use **Strong** for a behavior repeatedly verified in its stated scope, **Good** for a useful but narrow pass, **Limited** for partial coverage, **Weak** for a concrete gap, and **Not implemented** where no working path or measurement exists. A rating never expands the five-row evidence base.

## Area ratings

- **Query correctness and parity — Strong for the tested queries; limited scope.** CSV/Python, SQLite 3.45.1, and DuckDB 2.0.1 returned identical canonical result hashes for all three tested queries. The benchmark rechecked output parity on every timed execution: 1,000 runs plus 25 warmups per query and engine. These checks cover summary totals, grouping by platform, and filtered run details only.

- **Import validation and idempotence — Good for the aggregate CSV path.** A fresh import inserted all five source rows. Twenty reimports inserted zero new rows and left the summary unchanged. Existing tests also cover rejection of invalid counts and transaction rollback. One correctness caveat remains: the importer derives `run_id` from the complete row contents, so two distinct real executions with identical CSV fields would collapse as a duplicate. The current file has no exact duplicate rows, but that does not establish the intended identity rule for future ledgers.

- **Import throughput — Measured, not capacity-qualified.** The initial import took 109.1 ms, or 45.8 rows/s over five rows; this includes schema initialization and CSV parsing. Twenty idempotent reimports had a 58.2 ms median and 65.0 ms p95. Fixed startup and connection costs dominate such a small sample, so these rates should not be extrapolated.

- **Single-query latency — Not sufficient to select an engine.** On this host, repeated medians across the three queries were 0.0298–0.0301 ms for CSV/Python, 0.0085–0.0112 ms for SQLite, and 0.7039–0.8741 ms for DuckDB. All three implementations were parity-checked, but the database schemas differ: SQLite contains the two normalized tables required by these queries, while DuckDB uses the broader project schema. The query set and five-row file do not represent a production workload.

- **Read concurrency — Good smoke-test behavior; no scaling conclusion.** The harness ran three rounds at each setting, with 250 identical aggregate queries per reader per round. Each reader used an independent read-only connection and one DuckDB thread. All results matched exactly and there were zero errors. Median aggregate throughput across the three rounds was 1,317.7 QPS at one reader, 2,438.6 at two, 1,650.5 at four, and 1,923.9 at eight. The corresponding medians of the round-level per-query p50/p95 latencies were 0.699/0.927 ms, 0.710/0.897 ms, 1.218/2.996 ms, and 4.069/5.658 ms. Throughput is non-monotonic and latency worsens at higher reader counts. Every reader repeatedly queried the same five-row database; this is only a local small-file read smoke test. Writer concurrency was deliberately not tested because the design requires one writer process and the plugin has no worker-spool/collector path.

- **Privacy and redaction — Weak; close the gaps before broader capture.** Three of six synthetic `redact_text` canaries remained: an `Authorization: Bearer ...` header, `token=...`, and a `ghp_...`-like value. The `api_key=...`, bare `Bearer ...`, and `password=...` samples were redacted. These booleans were recorded without persisting the canary samples. A separate isolated temporary-CSV probe confirmed that the importer stores a synthetic `token=...` canary in the `command` column unchanged; this was a privacy probe, not a project telemetry row. Code inspection also shows that only notes are passed through the redactor during CSV import, while command and platform are stored directly. Test and failure identifiers use deterministic unsalted SHA-256 hashes: they avoid storing names/text verbatim but remain linkable pseudonyms, not anonymization.

- **Pytest capture — Limited positive evidence.** The focused subprocess test exercised one pass, one failure with a seeded API-key-like value, and one skip. It confirmed those outcomes were recorded and that the test name, skip reason, and seeded value did not appear in the inspected temporary database. The plugin holds all per-test results in memory until session finish, then writes one transaction. A process interruption before that write can lose the buffered run. Retries overwrite phase entries and are ultimately written as attempt 1; collection errors and interrupted-session records are not captured. The global single-config implementation also has no pytest-xdist spool/collector support.

- **Closed-file copy/restore — Good for a basic integrity check; backup system not implemented.** Copying the closed DuckDB file to a temporary location and reopening it read-only produced the same summary and view-result hash. This does not test a live/atomic backup, scheduled backups, crash recovery, retention, or rebuild from portable exports.

- **Schema completeness and migration — Limited foundation.** The v1 schema defines runs, summaries, per-test results, pipeline events, measurements, LLM usage, artifact references, and query benchmarks. The actual imported ledger populated only five runs and five summaries; per-test, event, measurement, LLM, artifact, and benchmark tables were empty. Schema version 1 was verified, but there is no upgrade/downgrade migration path or migration test.

- **LLM tokenomics and portable analytics — Not implemented or measured.** The schema has an LLM-usage table and a token-efficiency view, but this ledger contains no LLM usage. There was no provider-token reconciliation, quality-per-token test, LLM query benchmark, Parquet export/rebuild, or comparison with PostgreSQL or ClickHouse. No speed, token, or database-superiority claim is supported.

## Prioritized improvements

1. **Close privacy gaps first.** The local draft below fixes the observed authorization, token-assignment, provider-prefix, and raw command-field misses. Before broader capture, review escaped/encoded and multiline inputs, event attributes, failure representations, and any additional free-text fields. Prefer allowlisted structured metadata over arbitrary command text. Decide whether identifiers need keyed HMACs, rotation, or retention controls; do not describe plain hashes as anonymization.
2. **Make capture durable and process-safe.** Implement per-worker append-only JSONL spools with a single collector, or another explicitly supported writer architecture. Record retry attempts distinctly, and capture collection failures, cancellations, and interrupted sessions. Test duplicate delivery, partial writes, crashes during flush, and recovery without silently losing records.
3. **Define run identity and evolve the schema safely.** Use an upstream run/execution identifier where available. If only CSV exists, preserve source identity and row position so separate identical executions are not accidentally deduplicated while repeated file delivery remains idempotent. Add versioned migration fixtures and tests for upgrades from every supported version.
4. **Add operational backup and portability paths.** Test an atomic/consistent backup while the database is in use, automated restore verification, retention, and deterministic export/rebuild from JSONL or Parquet. Keep large logs and artifacts outside the database and store controlled references.
5. **Grow real telemetry before tuning.** Integrate actual stage events, per-test results, measurements, artifacts, and provider-reported LLM usage. Accumulate authentic project runs; do not manufacture ledger rows. Then preregister representative queries and measure cold/warm behavior, ingest, p50/p95, CPU/RSS, database size, and concurrency across equivalent schemas.
6. **Qualify token claims separately.** Compare deterministic filtered SQL and any LLM-assisted workflow against the same unfiltered-context baseline. Count authoritative input/output/cached tokens and cost per correct answer, with answer exactness and data coverage reported. The current schema alone is not evidence of token savings.

## Follow-up draft patch — 2026-11-22

A local, uncommitted patch extends redaction to authorization headers, `token` and related assignments, prefixed environment-variable names, `--flag value` arguments, quoted and JSON-like values, bearer credentials, and common GitHub/GitLab/Slack/Google token prefixes. CSV imports now pass `command` and `platform` through the same redactor as `notes`. Regression tests inspect persisted values and include the earlier canaries plus escaped quoted values. The focused telemetry tests pass (15 tests), and Ruff passes. This is a draft fix for the identified cases, not proof that all credential formats or telemetry fields are safe; the original ratings and JSON describe the assessment before this follow-up.

## Reproduction and artifacts

From the repository root, with the optional DuckDB dependency installed, run:

```bash
PYTHONPATH=src python scripts/assess_telemetry_system.py \
  --csv testing/results.csv \
  --repeated-imports 20 \
  --queries-each-reader 250 \
  --reader-rounds 3 \
  --output benchmarks/results/telemetry-system-assessment-2026-11-22.json

PYTHONPATH=src python scripts/benchmark_telemetry_engines.py \
  --csv testing/results.csv \
  --repetitions 1000 \
  --warmups 25 \
  --output benchmarks/results/telemetry-ledger-engine-benchmark-2026-11-22.json
```

The [machine-readable system-assessment results](../benchmarks/results/telemetry-system-assessment-2026-11-22.json) contain the individual concurrency rounds, import timings, source digest, schema counts, and redaction-canary outcomes. The [parity benchmark report](telemetry-engine-benchmark-2026-11-22.md) and [its JSON measurements](../benchmarks/results/telemetry-ledger-engine-benchmark-2026-11-22.json) contain per-query engine timings and output hashes. The [assessment harness](../scripts/assess_telemetry_system.py) and [benchmark harness](../scripts/benchmark_telemetry_engines.py) use temporary databases; they do not modify the local pilot database.
