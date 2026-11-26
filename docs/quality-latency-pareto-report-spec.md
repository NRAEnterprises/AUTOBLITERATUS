# Quality-versus-latency Pareto report: CLI implementation specification

**Status:** Implemented in the read-only telemetry query pack and CLI. The query wrapper fetches at most `MAX_INPUT_OBSERVATIONS + 1` matching records to detect and reject oversized inputs; wrapper and CLI end-to-end tests cover filters, frontier output, and non-mutation.

## 1. Purpose

Add a bounded, read-only telemetry CLI report that compares quality and latency across explicitly identified pipeline/model configurations. For each comparable cohort, report the observed candidates on the Pareto frontier: a candidate is efficient when no other candidate has at least as good quality and no worse latency, with a strict improvement in at least one objective.

This is an observational comparison, not proof that a candidate caused an improvement or that small differences are statistically meaningful. Do not collapse quality and latency into a single weighted score. The report must preserve sample counts and provenance so users can judge the evidence.

## 2. Existing data and boundaries

Use the v1 `measurements` table, joined to `pipeline_runs` for run-date and commit provenance. A measurement has a run ID, stage, metric name, numeric value, unit, timestamp, and JSON dimensions. The quality-outcome importer already writes numeric outcomes into this table and preserves identical repeated observations as distinct records. Latency should also be recorded as a numeric measurement in the same table.

No schema migration is required for the first implementation. The report must not infer a missing quality result, latency, unit, candidate identity, or cohort identity. It must not mutate data, fetch artifacts, call an LLM, or emit causal or statistical-significance claims.

For results to be comparable, the telemetry producers must record stable dimensions on both metric types. Typical cohort dimensions are `workload_id`, `model_id`, `model_revision`, and `dataset_id`. Typical candidate dimensions are `backend`, `precision`, `quantization`, `batch_size`, or an explicit `candidate_id`. Hardware may be a cohort dimension when controlling for hardware, or a candidate dimension when comparing hardware; the caller must choose deliberately. Candidate dimensions must identify a configuration sufficiently to avoid combining different treatments.

## 3. CLI interface

Add the `quality-latency-pareto` subcommand to `scripts/telemetry_db.py`:

```text
python scripts/telemetry_db.py [--db DB_PATH] quality-latency-pareto \
  --quality-metric METRIC \
  --latency-metric METRIC \
  --cohort-by KEY[,KEY...] \
  --candidate-by KEY[,KEY...] \
  [--quality-stage STAGE] \
  [--latency-stage STAGE] \
  [--quality-direction higher|lower] \
  [--start-date YYYY-MM-DD] \
  [--end-date YYYY-MM-DD] \
  [--min-paired-runs N] \
  [--limit N]
```

Arguments:

- `--quality-metric` and `--latency-metric` are required exact names from `measurements.metric`; they must differ.
- `--cohort-by` is a required comma-separated, non-empty set of dimensions that define observations that may be compared with one another. Every key must be present and non-null in the matched measurement records.
- `--candidate-by` is a required comma-separated, non-empty set of dimensions that distinguish the candidates being compared. Every key must be present and non-null. Keys must be unique within each list and disjoint between the two lists.
- Dimension names must match `^[A-Za-z_][A-Za-z0-9_]*$`, consistent with the existing measurement-summary filter. They are data keys, not SQL fragments; SQL generation must validate them and use bound JSON paths/values wherever possible. Never accept arbitrary SQL expressions.
- `--quality-stage` and `--latency-stage` optionally restrict each metric to a specific stage. If omitted and the selected metric is recorded at multiple stages in the date-filtered input, fail with an actionable error requiring the corresponding stage argument; do not silently combine stage semantics.
- `--quality-direction` defaults to `higher`; allow `lower` for loss/error metrics. Latency is always minimized.
- `--start-date` and `--end-date` follow the existing inclusive ISO `YYYY-MM-DD` convention and filter on each measurement's `measured_at` date. Reject an inverted date range.
- `--min-paired-runs` defaults to `1`, must be a positive integer, and filters candidate points by the number of distinct paired run IDs. Keep the default permissive for early sparse telemetry, but show the resulting sample count prominently. Do not describe a one-run point as robust evidence.
- `--limit` follows existing query-pack behavior: default `200`, positive integer, hard cap `1000`. Report whether output was truncated.

The command is invoked after the global `--db` option, consistent with the current CLI. It emits JSON to stdout in the existing query-report envelope; errors go to stderr and return a non-zero exit code.

## 4. Pairing and aggregation rules

### 4.1 Record selection

Select non-null numeric values for the two requested metrics, subject to date and stage filters. Within each selected metric, retain only rows whose required cohort and candidate dimensions exist, are non-null JSON scalars, and pass the report's safe-output checks. Missing, nested, or unsafe dimension values are excluded and counted in coverage metadata; never print untrusted dimension payloads.

Check that each metric has one consistent unit within each cohort. If candidates in the same cohort use incompatible units for either metric, fail with a clear validation error rather than comparing unlike values. Return both metric units in each cohort's metadata or rows.

### 4.2 Matched run observations

Pair metrics only on the exact composite key:

```text
run_id + all cohort dimension values + all candidate dimension values
```

Do not pair measurements across different runs. For each key, compute the median of the quality observations and median of the latency observations independently. This collapses repeated measurements within a run to one paired run-level point while retaining their raw counts.

A key with only quality or only latency is unpaired and excluded from Pareto calculations; count it in metadata. Do not impute the missing side or treat it as zero.

### 4.3 Candidate points

Within each cohort and candidate key, calculate:

- `quality_value`: median of the paired run-level quality medians;
- `latency_value`: median of the paired run-level latency medians;
- `paired_runs`: number of distinct run IDs contributing both metrics;
- `quality_measurements` and `latency_measurements`: total raw selected records contributing to the matched runs, retaining the two counts separately;
- `date_start` and `date_end`: earliest and latest measurement dates contributing to the point;
- `commits`: sorted distinct non-null commit SHAs for those paired runs.

A candidate is omitted if `paired_runs < min_paired_runs`; filtered candidate counts are included in coverage metadata. Use median aggregation for resistance to outliers. Do not weight runs by their number of repeated measurements, since that would let a heavily sampled run dominate the candidate summary.

### 4.4 Pareto classification

Compare candidate points only within an identical cohort. Candidate A dominates candidate B if:

- for `higher` quality, `A.quality_value >= B.quality_value`; or for `lower` quality, `A.quality_value <= B.quality_value`;
- `A.latency_value <= B.latency_value`; and
- at least one of those inequalities is strict.

Set `pareto_efficient` to false when any other point dominates the candidate, and true otherwise. Equal points do not strictly dominate one another; retain all tied candidates as efficient. Return a deterministically sorted `dominated_by` list containing the candidate-dimension objects of all strict dominators. Use exact stored numeric values for v1 comparisons; do not silently introduce rounding or a tolerance. Output values may be rounded for display only if the unrounded values remain available in machine-readable output.

## 5. Output contract

Return one row per eligible candidate point, including both efficient and dominated points, in a `QueryReport`-style envelope:

```json
{
  "name": "quality_latency_pareto",
  "rows": [
    {
      "cohort": {"workload_id": "task-a", "model_id": "model-x"},
      "candidate": {"backend": "cpu", "precision": "fp32"},
      "quality_metric": "answer_quality",
      "quality_unit": "score",
      "quality_direction": "higher",
      "quality_value": 0.91,
      "latency_metric": "latency_ms",
      "latency_unit": "ms",
      "latency_value": 120.0,
      "paired_runs": 4,
      "quality_measurements": 8,
      "latency_measurements": 12,
      "pareto_efficient": true,
      "dominated_by": [],
      "date_start": "2026-11-20",
      "date_end": "2026-11-25",
      "commits": ["abc123"]
    }
  ],
  "metadata": {
    "source_tables": ["measurements", "pipeline_runs"],
    "filters": {},
    "coverage": {},
    "provenance": {}
  }
}
```

This is an illustrative shape, not a fixture with assumed production values. `metadata.filters` must echo metrics, stages, dimension keys, direction, date bounds, minimum paired runs, requested limit, and applied limit. `metadata.coverage` must include selected quality/latency records, records excluded for invalid/missing dimensions, paired run-candidate keys, unpaired keys, eligible candidates, candidates filtered by minimum runs, returned rows, and truncation status. `metadata.provenance` must include contributing date bounds and distinct commit SHAs. Empty results are valid and should return the same envelope with empty rows and zero coverage counts.

Sort deterministically by canonical cohort JSON, then efficient candidates before dominated candidates, then canonical candidate JSON. Preserve original dimension scalar types in JSON output; do not stringify values merely to make them sortable.

## 6. Safety and query constraints

- Open DuckDB in read-only mode and use fixed, reviewed SQL templates.
- Bind metric names, stage names, dates, limits, and JSON paths as parameters. Validate dimension-key syntax before constructing any dynamic grouping expression; never interpolate user-provided values into SQL.
- Enforce the existing 1,000-row maximum and keep intermediate work bounded by selected rows/date filters and explicit cohort/candidate keys. If a separate input-row cap is required for predictable resource use, define and test it before implementation and return an explicit error rather than silently truncating source evidence.
- Check generated dimension values for safe scalar type, length, and credential-like content before returning them. Do not include raw notes, messages, URIs, or arbitrary JSON blobs in output.
- Preserve missingness; report the number excluded for incomplete pairing or dimensions.
- A report must carry sufficient filters, unit, count, commit, and date provenance to reproduce the comparison. No LLM-generated SQL is part of this command.

## 7. Implementation record

1. The CLI parses comma-separated dimension keys, rejects empty, duplicate, unsafe, and overlapping keys, and validates positive limits and minimum paired-run counts.
2. `quality_latency_pareto_report` in `src/autobliteratus/observability/query_pack.py` uses fixed SQL with bound metric, stage, date, and limit values; opens DuckDB in read-only mode; fetches at most `MAX_INPUT_OBSERVATIONS + 1`; and raises rather than silently discarding source observations when the cap is exceeded.
3. `quality-latency-pareto` in `scripts/telemetry_db.py` exposes the specified filters and serializes the engine's JSON-ready report envelope.
4. The observability roadmap and this spec document the pairing, medians, Pareto rule, sample/provenance fields, and non-causal interpretation limits.
5. Pure-engine, wrapper, and CLI tests cover expected frontier membership, query filters and read-only access, explicit input-cap failure, parser errors, and unchanged database row counts.

## 8. Required tests and acceptance criteria

Tests must cover at least:

- a known three-or-more-candidate frontier, including one dominated point;
- both quality directions (`higher` and `lower`);
- a latency-quality tradeoff where neither candidate dominates the other;
- tied points, ensuring strict dominance semantics preserve ties;
- separate cohort values never dominating across cohorts;
- disjoint cohort/candidate dimensions and rejection of repeated/invalid keys;
- run-level pairing only, including unpaired quality-only and latency-only runs;
- repeated measurements aggregated to per-run medians, then median across runs without weighting by repetition count;
- `min_paired_runs` filtering and explicit coverage counts;
- null values, missing dimensions, nested dimension objects, unsafe output values, and inconsistent units;
- explicit stages and rejection of ambiguous multi-stage metrics;
- inclusive date filters, invalid date range, deterministic output ordering, empty results, limit cap/truncation metadata;
- SQL-injection-shaped dimension keys/metric/stage values, confirming validation and bound parameters;
- provenance containing the correct dates/commits and no unrelated cohort records;
- a CLI end-to-end test over initialized DuckDB data, checking exact JSON fields and exit behavior.

Acceptance requires exact expected frontier membership and values on deterministic fixtures; no cross-run pairing; no imputation; no data writes; passing focused tests, Ruff, compilation, and the full observability regression suite. If the broader repository suite cannot collect because optional model dependencies are unavailable, record that limitation separately rather than claiming a full-suite pass.

## 9. Known interpretation limits

The v1 schema has no stable experiment ID, repetition ID, standardized device inventory, or direct LLM-usage-to-accepted-result relationship. The report therefore treats each run plus selected dimension values as the pairing unit, depends on producers to record consistent dimensions on both metrics, and cannot estimate within-run covariance or statistical confidence intervals. Multiple commits/configurations may contribute to a candidate point; the output lists commits and time range but does not attribute causality. Add richer pairing identifiers, device/model metadata, uncertainty estimates, or acceptance/cost linkage only with explicit schema/producer contracts and versioned tests.
